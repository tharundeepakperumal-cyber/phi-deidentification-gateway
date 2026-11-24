import argparse
import json
import math
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.evaluate_quantization import (
    MODEL_PATH,
    load_data,
    predict_model,
)


def percentile(values, p):
    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return ordered[index]


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
                current = self.process.memory_info().rss / (1024 ** 2)
                self.peak_mb = max(self.peak_mb, current)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8"],
        required=True,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    mode = args.mode

    torch.set_num_threads(1)

    print("\n========== QUANTIZATION BENCHMARK ==========")
    print("Mode:", mode.upper())
    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())

    sampler = MemorySampler()
    sampler.start()

    process = psutil.Process()
    baseline_rss = sampler.current_mb()

    # Load tokenizer
    tokenizer_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True,
    )

    tokenizer_load_seconds = (
        time.perf_counter() - tokenizer_start
    )

    # Load FP32 model
    model_start = time.perf_counter()

    model = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_PATH),
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    model.to("cpu")
    model.eval()

    model_load_seconds = time.perf_counter() - model_start

    quantization_seconds = 0.0
    converted_layers = 0

    if mode == "int8":
        print("\nSelecting QNNPACK engine")
        torch.backends.quantized.engine = "qnnpack"

        from torch.ao.quantization import quantize_dynamic

        quantization_start = time.perf_counter()

        model = quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
            inplace=True,
        )

        quantization_seconds = (
            time.perf_counter() - quantization_start
        )

        converted_layers = sum(
            1
            for module in model.modules()
            if isinstance(
                module,
                torch.ao.nn.quantized.dynamic.Linear,
            )
        )

    loaded_rss = sampler.current_mb()

    print("\n========== MODEL LOADING ==========")
    print("Tokenizer load:", round(tokenizer_load_seconds, 3), "sec")
    print("FP32 model load:", round(model_load_seconds, 3), "sec")
    print("Quantization:", round(quantization_seconds, 3), "sec")
    print("INT8 Linear layers:", converted_layers)

    # Load validation records
    records = load_data()
    texts = [record["text"] for record in records]

    print("\nValidation records:", len(texts))
    print("Warmup records: 20")
    print("Measured passes:", args.repeats)

    # Warm up both tokenizer and model.
    for text in texts[:20]:
        predict_model(text, tokenizer, model)

    print("\nWarmup completed.")

    # Measure full per-record inference, including tokenization.
    latencies_ms = []
    measured_start = time.perf_counter()

    for repeat in range(args.repeats):
        for index, text in enumerate(texts, start=1):
            start = time.perf_counter()

            predict_model(text, tokenizer, model)

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        print(
            f"Completed pass {repeat + 1}/{args.repeats}"
        )

    total_seconds = time.perf_counter() - measured_start
    total_records = len(latencies_ms)

    sampler.stop()

    final_rss = sampler.current_mb()
    sampled_peak_rss = sampler.peak_mb

    print("\n========== PERFORMANCE RESULTS ==========")
    print("Total measured records:", total_records)
    print("Total inference time:", round(total_seconds, 3), "sec")
    print(
        "Throughput:",
        round(total_records / total_seconds, 2),
        "records/sec",
    )
    print("P50 latency:", round(percentile(latencies_ms, 0.50), 3), "ms")
    print("P95 latency:", round(percentile(latencies_ms, 0.95), 3), "ms")
    print("Mean latency:", round(statistics.mean(latencies_ms), 3), "ms")

    print("\n========== MEMORY RESULTS ==========")
    print("Baseline RSS:", round(baseline_rss, 2), "MB")
    print("RSS after model load:", round(loaded_rss, 2), "MB")
    print("Final RSS:", round(final_rss, 2), "MB")
    print("Sampled peak RSS:", round(sampled_peak_rss, 2), "MB")
    print(
        "RSS increase after load:",
        round(loaded_rss - baseline_rss, 2),
        "MB",
    )

    report = {
        "mode": mode,
        "device": "cpu",
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "validation_records": len(texts),
        "repeats": args.repeats,
        "measured_records": total_records,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_load_seconds": model_load_seconds,
        "quantization_seconds": quantization_seconds,
        "quantized_linear_layers": converted_layers,
        "total_inference_seconds": total_seconds,
        "throughput_records_per_second": total_records / total_seconds,
        "p50_latency_ms": percentile(latencies_ms, 0.50),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "baseline_rss_mb": baseline_rss,
        "rss_after_model_load_mb": loaded_rss,
        "final_rss_mb": final_rss,
        "sampled_peak_rss_mb": sampled_peak_rss,
    }

    report_dir = Path("reports/template_aware")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"benchmark_{mode}.json"
    report_path.write_text(json.dumps(report, indent=4))

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    main()
