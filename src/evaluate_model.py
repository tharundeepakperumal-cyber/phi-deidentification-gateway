import json
from collections import defaultdict
from pathlib import Path

import torch

from src.model_inference import (
    load_model,
    extract_entities,
    CORRECTION_PATTERNS,
    DEVICE,
)


DATASET_PATH = Path("data/raw/synthetic_clinical_notes.json")
REPORT_DIR = Path("reports/template_aware")
JSON_REPORT = REPORT_DIR / "entity_evaluation.json"
TEXT_REPORT = REPORT_DIR / "entity_evaluation.txt"

VALIDATION_TEMPLATE_IDS = {6, 7}

LABEL_MAP = {
    "NAME": "PERSON",
    "PERSON": "PERSON",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "MRN": "MRN",
}

LABELS = ["PERSON", "PHONE", "EMAIL", "MRN"]

MAX_LENGTH = 512
STRIDE = 64


def load_validation_data():
    """Load only the held-out validation templates."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with DATASET_PATH.open(encoding="utf-8") as file:
        records = json.load(file)

    validation = [
        record for record in records
        if record["template_id"] in VALIDATION_TEMPLATE_IDS
    ]

    if len(validation) != 250:
        raise ValueError(
            f"Expected 250 validation records, got {len(validation)}"
        )

    return validation


def get_gold_entities(text, annotations):
    """Extract every occurrence of each annotated entity."""

    gold = set()

    for original_label, value in annotations.items():

        label = LABEL_MAP.get(original_label)

        if label is None:
            continue

        if not value:
            continue

        start = 0
        found = False

        while True:
            start = text.find(value, start)

            if start == -1:
                break

            gold.add((label, start, start + len(value)))
            start += len(value)
            found = True

        if not found:
            raise ValueError(
                f"Annotation not found: {original_label}={value!r}"
            )

    return gold


def predict_model_only(text, tokenizer, model):
    """Run DistilBERT sequentially without regex corrections."""

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )

    offsets = encoding.pop("offset_mapping")
    encoding.pop("overflow_to_sample_mapping", None)

    entities = set()

    # Process one overflow chunk at a time.
    for chunk_index in range(offsets.shape[0]):

        chunk_inputs = {
            key: value[chunk_index:chunk_index + 1].to(DEVICE)
            for key, value in encoding.items()
        }

        with torch.inference_mode():
            outputs = model(**chunk_inputs)

            chunk_predictions = torch.argmax(
                outputs.logits[0],
                dim=-1,
            ).cpu().tolist()

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
            chunk_predictions,
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

            if entity_type not in {
                "NAME", "PHONE", "EMAIL", "MRN"
            }:
                flush()
                continue

            if prefix == "B":
                flush()

                current = {
                    "label": LABEL_MAP[entity_type],
                    "start": start,
                    "end": end,
                }

            elif prefix == "I":

                if (
                    current is None
                    or current["label"] != LABEL_MAP[entity_type]
                ):
                    flush()

                    current = {
                        "label": LABEL_MAP[entity_type],
                        "start": start,
                        "end": end,
                    }

                else:
                    current["end"] = end

            else:
                flush()

        flush()

        # Release chunk-specific inference outputs.
        del outputs
        del chunk_inputs

    return entities

def predict_regex_only(text):
    """Run deterministic regex detection without DistilBERT."""

    entities = set()

    for entity_type, pattern in CORRECTION_PATTERNS.items():

        for match in pattern.finditer(text):

            entities.add((
                entity_type,
                match.start(),
                match.end(),
            ))

    return entities


def predict_combined(text):
    """Run the complete production inference pipeline."""

    predictions = extract_entities(text)

    return {
        (
            LABEL_MAP[entity["label"]],
            entity["start"],
            entity["end"],
        )
        for entity in predictions
        if entity["label"] in LABEL_MAP
    }


def calculate_metrics(gold_entities, predicted_entities):
    """Calculate exact-span entity-level metrics."""

    counts = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for label in LABELS:

        gold = {
            entity for entity in gold_entities
            if entity[1] == label
        }

        predicted = {
            entity for entity in predicted_entities
            if entity[1] == label
        }

        counts[label]["tp"] = len(gold & predicted)
        counts[label]["fp"] = len(predicted - gold)
        counts[label]["fn"] = len(gold - predicted)

    results = {}
    total_tp = total_fp = total_fn = 0

    for label in LABELS:

        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    results["OVERALL_MICRO"] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }

    return results


def main():

    print("\n========== PHI GUARDIAN ENTITY EVALUATION ==========")

    records = load_validation_data()

    print("Validation records:", len(records))
    print("Validation templates:", sorted(VALIDATION_TEMPLATE_IDS))
    print("Device:", DEVICE)

    tokenizer, model = load_model()
    model.eval()

    gold_all = set()
    model_all = set()
    regex_all = set()
    combined_all = set()

    for index, record in enumerate(records, start=1):

        text = record["text"]

        gold = get_gold_entities(
            text,
            record["entities"],
        )

        model_predictions = predict_model_only(
            text,
            tokenizer,
            model,
        )

        regex_predictions = predict_regex_only(text)

        combined_predictions = predict_combined(text)

        gold_all.update(
            (index, *entity) for entity in gold
        )

        model_all.update(
            (index, *entity) for entity in model_predictions
        )

        regex_all.update(
            (index, *entity) for entity in regex_predictions
        )

        combined_all.update(
            (index, *entity) for entity in combined_predictions
        )

        if index % 25 == 0:
            print(f"Evaluated {index}/{len(records)} notes")

    model_results = calculate_metrics(gold_all, model_all)
    regex_results = calculate_metrics(gold_all, regex_all)
    combined_results = calculate_metrics(gold_all, combined_all)

    report = {
        "dataset": str(DATASET_PATH),
        "evaluation_type": "Exact character-span entity evaluation",
        "validation_records": len(records),
        "validation_template_ids": sorted(VALIDATION_TEMPLATE_IDS),
        "model": model_results,
        "regex": regex_results,
        "combined_pipeline": combined_results,
        "notes": [
            "Only validation templates 6 and 7 were evaluated.",
            "Metrics use exact entity type and character-span matching.",
            "Model-only predictions exclude regex corrections.",
            "Regex-only predictions exclude DistilBERT.",
            "Combined results use the production extract_entities pipeline.",
            "Validation data was also used for model selection during training; "
            "these results are not independent test-set results.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    with TEXT_REPORT.open("w", encoding="utf-8") as file:

        file.write("PHI GUARDIAN ENTITY EVALUATION\n")
        file.write("=" * 55 + "\n")
        file.write(f"Validation records: {len(records)}\n")
        file.write("Validation templates: 6, 7\n")
        file.write("Matching: Exact entity type and character span\n")

        for name, results in [
            ("DISTILBERT ONLY", model_results),
            ("REGEX ONLY", regex_results),
            ("COMBINED PIPELINE", combined_results),
        ]:

            file.write("\n" + "=" * 55 + "\n")
            file.write(name + "\n")
            file.write("=" * 55 + "\n")

            for label, metrics in results.items():

                file.write(f"\n{label}\n")

                for metric, value in metrics.items():
                    file.write(f"  {metric}: {value}\n")

        file.write("\nEvaluation completed.\n")

    print("\n========== EVALUATION RESULTS ==========")

    for name, results in [
        ("DISTILBERT ONLY", model_results),
        ("REGEX ONLY", regex_results),
        ("COMBINED PIPELINE", combined_results),
    ]:

        print(f"\n{name}")

        for label, metrics in results.items():

            print(
                f"{label}: "
                f"TP={metrics['tp']} "
                f"FP={metrics['fp']} "
                f"FN={metrics['fn']} "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f}"
            )

    print("\nEvaluation completed.")
    print("JSON report:", JSON_REPORT)
    print("Text report:", TEXT_REPORT)


if __name__ == "__main__":
    main()
