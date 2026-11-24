import argparse
import hashlib
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.ao.quantization import quantize_dynamic

# Ensure the project root is importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_quantization import MODEL_PATH, load_data, predict_model


def percentile(values, p):
    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return ordered[index]


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


class DeviceModel:
    """Move evaluation inputs to the selected device."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.config = model.config

    def __call__(self, **inputs):
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }
        return self.model(**inputs)


class MemorySampler:
    def __init__(self, interval=0.01):
        self.process = psutil.Process()
        self.interval = interval
        self.peak_mb = 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                rss = self.process.memory_info().rss / (1024 ** 2)
                self.peak_mb = max(self.peak_mb, rss)
            except psutil.Error:
                pass
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def current_mb(self):
        return self.process.memory_info().rss / (1024 ** 2)


def mps_memory():
    if not torch.backends.mps.is_available():
        return {}

    result = {}

    for name in (
        "current_allocated_memory",
        "driver_allocated_memory",
    ):
        function = getattr(torch.mps, name, None)
        if function is not None:
            try:
                result[name + "_mb"] = function() / (1024 ** 2)
            except Exception:
                result[name + "_mb"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP32 or INT8 inference on CPU or FP32 on Apple MPS."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        required=True,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8"],
        default="fp32",
    )
    args = parser.parse_args()

    if args.mode == "int8" and args.device != "cpu":
        parser.error("INT8 dynamic quantization is supported on CPU only")

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    device = torch.device(args.device)

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    torch.set_num_threads(1)

    print(f"\n========== {args.mode.upper()} DEVICE BENCHMARK ==========")
    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())

    sampler = MemorySampler()
    sampler.start()

    try:
        baseline_rss = sampler.current_mb()

        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            use_fast=True,
        )
        tokenizer_load_seconds = time.perf_counter() - start

        start = time.perf_counter()
        model = AutoModelForTokenClassification.from_pretrained(
            str(MODEL_PATH),
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        synchronize(device)
        model_load_seconds = time.perf_counter() - start
        model_loaded_rss = sampler.current_mb()

        quantization_seconds = 0.0
        converted = 0

        if args.mode == "int8":
            if "qnnpack" not in torch.backends.quantized.supported_engines:
                raise RuntimeError("QNNPACK backend is unavailable")

            torch.backends.quantized.engine = "qnnpack"

            quantization_start = time.perf_counter()

            model = quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

            quantization_seconds = (
                time.perf_counter() - quantization_start
            )

            converted = sum(
                isinstance(
                    module,
                    torch.ao.nn.quantized.dynamic.Linear,
                )
                for module in model.modules()
            )

            print("INT8 Linear layers:", converted)

            if converted == 0:
                raise RuntimeError("No Linear layers were quantized")

        # Capture memory after optional INT8 quantization.
        post_quantization_rss = sampler.current_mb()

        device_memory_after_load = mps_memory()

        predictor_model = DeviceModel(model, device)

        records = load_data()
        texts = [record["text"] for record in records]

        if not texts:
            raise RuntimeError("Validation dataset is empty")

        print("Validation records:", len(texts))
        print("Warmup records:", min(20, len(texts)))
        print("Measured passes:", args.repeats)

        # Warmup.
        for text in texts[:20]:
            predict_model(text, tokenizer, predictor_model)

        synchronize(device)

        print("Warmup completed.")

        warmup_rss = sampler.current_mb()

        latencies_ms = []
        pass_signatures = []
        first_pass_entity_count = None

        # Measured benchmark.
        measured_start = time.perf_counter()

        for repeat in range(args.repeats):
            pass_predictions = []

            for text in texts:
                synchronize(device)

                start = time.perf_counter()

                predictions = predict_model(
                    text,
                    tokenizer,
                    predictor_model,
                )

                synchronize(device)

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                latencies_ms.append(elapsed_ms)
                pass_predictions.append(sorted(predictions))

            serialized = json.dumps(
                pass_predictions,
                separators=(",", ":"),
            )

            signature = hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest()

            pass_signatures.append(signature)

            if repeat == 0:
                first_pass_entity_count = sum(
                    len(items) for items in pass_predictions
                )

            print(
                f"Completed pass {repeat + 1}/{args.repeats}"
            )

        synchronize(device)

        total_seconds = time.perf_counter() - measured_start

        measured_records = len(latencies_ms)

        final_rss = sampler.current_mb()
        peak_rss = sampler.peak_mb
        device_memory_final = mps_memory()

        prediction_signature = pass_signatures[0]

        prediction_consistent = (
            len(set(pass_signatures)) == 1
        )

    finally:
        sampler.stop()

    throughput = measured_records / total_seconds

    report = {
        "mode": args.mode,
        "device": args.device,
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "validation_records": len(texts),
        "repeats": args.repeats,
        "warmup_records": min(20, len(texts)),
        "measured_records": measured_records,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_load_seconds": model_load_seconds,
        "quantization_seconds": quantization_seconds,
        "quantized_linear_layers": converted,
        "total_inference_seconds": total_seconds,
        "throughput_records_per_second": throughput,
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "first_pass_predicted_entities": first_pass_entity_count,
        "prediction_signature": prediction_signature,
        "pass_prediction_signatures": pass_signatures,
        "prediction_consistent_across_passes": prediction_consistent,
        "baseline_rss_mb": baseline_rss,
        "rss_after_model_load_mb": model_loaded_rss,
        "rss_after_quantization_mb": post_quantization_rss,
        "rss_after_warmup_mb": warmup_rss,
        "final_rss_mb": final_rss,
        "sampled_peak_process_rss_mb": peak_rss,
        "device_memory_after_load": device_memory_after_load,
        "device_memory_final": device_memory_final,
    }

    report_dir = PROJECT_ROOT / "reports" / "template_aware"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        report_dir /
        f"benchmark_{args.mode}_{args.device}_device_final.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n========== PERFORMANCE RESULTS ==========")
    print("Device:", args.device.upper())
    print("Measured records:", measured_records)
    print("Throughput:", round(throughput, 2), "records/sec")
    print("P50 latency:", round(report["p50_latency_ms"], 3), "ms")
    print("P95 latency:", round(report["p95_latency_ms"], 3), "ms")
    print("Mean latency:", round(report["mean_latency_ms"], 3), "ms")

    print("\n========== MEMORY RESULTS ==========")
    print("Baseline RSS:", round(baseline_rss, 2), "MB")
    print("RSS after model load:", round(model_loaded_rss, 2), "MB")
    print("RSS after quantization:", round(post_quantization_rss, 2), "MB")
    print("RSS after warmup:", round(warmup_rss, 2), "MB")
    print("Final RSS:", round(final_rss, 2), "MB")
    print("Peak process RSS:", round(peak_rss, 2), "MB")
    print("Device memory after load:", device_memory_after_load)
    print("Device memory final:", device_memory_final)

    print("\nPrediction signature:", prediction_signature)
    print("Prediction consistent:", prediction_consistent)
    print("Pass signatures:", len(set(pass_signatures)), "unique")
    print("Predicted entities:", first_pass_entity_count)
    print("Report saved:", report_path)

if __name__ == "__main__":
    main()
