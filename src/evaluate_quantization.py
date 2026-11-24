
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from torch.ao.quantization import quantize_dynamic

from src.model_inference import (
    CORRECTION_PATTERNS,
    resolve_overlapping_entities,
)

MODEL_PATH = Path("models/template_aware")
DATASET_PATH = Path("data/raw/synthetic_clinical_notes_test.json")

TEST_TEMPLATE_IDS = {8, 9, 10, 11}
LABELS = ["PERSON", "PHONE", "EMAIL", "MRN"]

LABEL_MAP = {
    "NAME": "PERSON",
    "PERSON": "PERSON",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "MRN": "MRN",
}

torch.set_num_threads(1)
torch.backends.quantized.engine = "qnnpack"


def load_data():
    with DATASET_PATH.open(encoding="utf-8") as f:
        records = json.load(f)

    records = [
        r for r in records
        if r["template_id"] in TEST_TEMPLATE_IDS
    ]

    if len(records) != 200:
        raise ValueError(
            f"Expected 200 records, got {len(records)}"
        )

    return records


def load_gold(text, annotations):
    entities = set()

    for original_label, value in annotations.items():
        label = LABEL_MAP.get(original_label)

        if not label or not value:
            continue

        start = 0

        while True:
            start = text.find(value, start)

            if start == -1:
                break

            entities.add((
                label,
                start,
                start + len(value),
            ))

            start += len(value)

    return entities


def predict_model(text, tokenizer, model):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        stride=64,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )

    offsets = encoding.pop("offset_mapping")
    encoding.pop("overflow_to_sample_mapping", None)

    entities = set()

    for chunk_index in range(offsets.shape[0]):

        chunk_inputs = {
            key: value[chunk_index:chunk_index + 1]
            for key, value in encoding.items()
        }

        with torch.inference_mode():
            outputs = model(**chunk_inputs)

            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )[0].tolist()

        chunk_offsets = offsets[chunk_index].tolist()

        current = None

        def flush():
            nonlocal current

            if current is not None:
                entities.add((
                    current["label"],
                    current["start"],
                    current["end"],
                ))
                current = None

        for prediction, (start, end) in zip(
            predictions,
            chunk_offsets,
        ):

            if start == end:
                flush()
                continue

            label = model.config.id2label[prediction]

            if label == "O" or "-" not in label:
                flush()
                continue

            prefix, entity_type = label.split("-", 1)

            if entity_type not in LABEL_MAP:
                flush()
                continue

            mapped_label = LABEL_MAP[entity_type]

            if prefix == "B":
                flush()

                current = {
                    "label": mapped_label,
                    "start": start,
                    "end": end,
                }

            elif prefix == "I":

                if (
                    current is None
                    or current["label"] != mapped_label
                ):
                    flush()

                    current = {
                        "label": mapped_label,
                        "start": start,
                        "end": end,
                    }

                else:
                    current["end"] = end

            else:
                flush()

        flush()

        del outputs
        del chunk_inputs

    return entities


def predict_regex(text):
    entities = set()

    for label, pattern in CORRECTION_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.add((
                label,
                match.start(),
                match.end(),
            ))

    return entities


def predict_combined(text, model_entities):
    entities = [
        {
            "label": label,
            "start": start,
            "end": end,
            "text": text[start:end],
            "priority": 1,
        }
        for label, start, end in model_entities
    ]

    for label, pattern in CORRECTION_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({
                "label": label,
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "priority": 2,
            })

    resolved = resolve_overlapping_entities(entities)

    return {
        (
            LABEL_MAP.get(entity["label"], entity["label"]),
            entity["start"],
            entity["end"],
        )
        for entity in resolved
        if entity["label"] in LABEL_MAP
    }


def calculate_metrics(gold, predicted):
    counts = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for label in LABELS:
        g = {x for x in gold if x[1] == label}
        p = {x for x in predicted if x[1] == label}

        counts[label]["tp"] = len(g & p)
        counts[label]["fp"] = len(p - g)
        counts[label]["fn"] = len(g - p)

    tp = sum(counts[x]["tp"] for x in LABELS)
    fp = sum(counts[x]["fp"] for x in LABELS)
    fn = sum(counts[x]["fn"] for x in LABELS)

    results = {}

    for label in LABELS:
        a = counts[label]["tp"]
        b = counts[label]["fp"]
        c = counts[label]["fn"]

        precision = a / (a + b) if a + b else 0
        recall = a / (a + c) if a + c else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        results[label] = {
            "tp": a,
            "fp": b,
            "fn": c,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    results["OVERALL_MICRO"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }

    return results


def main():
    print("\n========== QUANTIZATION ACCURACY EVALUATION ==========")

    records = load_data()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True,
    )

    print("Independent test records:", len(records))
    print("Independent test templates:", sorted(TEST_TEMPLATE_IDS))

    print("\n========== LOAD FP32 CPU MODEL ==========")

    fp32 = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_PATH),
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    fp32.to("cpu")
    fp32.eval()

    print("\n========== CREATE INT8 MODEL ==========")

    int8 = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_PATH),
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    int8.to("cpu")
    int8.eval()

    int8 = quantize_dynamic(
        int8,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )

    converted = sum(
        isinstance(
            module,
            torch.ao.nn.quantized.dynamic.Linear,
        )
        for module in int8.modules()
    )

    print("INT8 Linear layers:", converted)

    gold_all = set()
    fp32_all = set()
    int8_all = set()

    fp32_combined_all = set()
    int8_combined_all = set()

    start = time.perf_counter()

    for index, record in enumerate(records, start=1):
        text = record["text"]

        gold = load_gold(text, record["entities"])

        fp32_predictions = predict_model(
            text, tokenizer, fp32
        )

        int8_predictions = predict_model(
            text, tokenizer, int8
        )

        fp32_combined = predict_combined(
            text, fp32_predictions
        )

        int8_combined = predict_combined(
            text, int8_predictions
        )

        gold_all.update(
            (index, *entity) for entity in gold
        )

        fp32_all.update(
            (index, *entity) for entity in fp32_predictions
        )

        int8_all.update(
            (index, *entity) for entity in int8_predictions
        )

        fp32_combined_all.update(
            (index, *entity) for entity in fp32_combined
        )

        int8_combined_all.update(
            (index, *entity) for entity in int8_combined
        )

        if index % 25 == 0:
            print(f"Evaluated {index}/{len(records)}")

    print(
        "\nEvaluation time:",
        round(time.perf_counter() - start, 2),
        "seconds",
    )

    results = {
        "FP32 CPU MODEL ONLY": calculate_metrics(
            gold_all, fp32_all
        ),
        "INT8 CPU MODEL ONLY": calculate_metrics(
            gold_all, int8_all
        ),
        "FP32 CPU + REGEX": calculate_metrics(
            gold_all, fp32_combined_all
        ),
        "INT8 CPU + REGEX": calculate_metrics(
            gold_all, int8_combined_all
        ),
    }

    for name, metrics in results.items():
        print("\n==========", name, "==========")

        for label, values in metrics.items():
            print(
                f"{label}: "
                f"TP={values['tp']} "
                f"FP={values['fp']} "
                f"FN={values['fn']} "
                f"P={values['precision']:.4f} "
                f"R={values['recall']:.4f} "
                f"F1={values['f1']:.4f}"
            )

    report_dir = Path("reports/template_aware")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "independent_test_quantization_accuracy.json"

    report = {
        "evaluation_type": "Exact character-span entity evaluation",
        "independent_test_records": len(records),
        "independent_test_templates": sorted(TEST_TEMPLATE_IDS),
        "quantized_linear_layers": converted,
        "results": results,
        "note": (
            "Independent synthetic test dataset using templates 8-11."
        ),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    main()
