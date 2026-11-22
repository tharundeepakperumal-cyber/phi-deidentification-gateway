import json
from collections import defaultdict
from pathlib import Path

from src.model_inference import extract_entities


DATASET_PATH = Path("data/raw/synthetic_clinical_notes.json")

LABEL_MAP = {
    "NAME": "PERSON",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "MRN": "MRN",
}

LABELS = ["PERSON", "PHONE", "EMAIL", "MRN"]


def main():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    total_notes = 0

    for record in data:
        text = record["text"]
        annotations = record["entities"]

        gold = set()

        for label, value in annotations.items():
            if label not in LABELS:
                continue

            start = text.find(value)

            if start == -1:
                raise ValueError(
                    f"Annotation not found: {label}={value!r}"
                )

            gold.add((label, start, start + len(value)))

        predictions = extract_entities(text)

        predicted = set()

        for entity in predictions:
            label = LABEL_MAP.get(entity["label"])

            if label is None:
                continue

            predicted.add(
                (label, entity["start"], entity["end"])
            )

        for label in LABELS:
            gold_for_label = {
                item for item in gold if item[0] == label
            }

            predicted_for_label = {
                item for item in predicted if item[0] == label
            }

            counts[label]["tp"] += len(
                gold_for_label & predicted_for_label
            )
            counts[label]["fp"] += len(
                predicted_for_label - gold_for_label
            )
            counts[label]["fn"] += len(
                gold_for_label - predicted_for_label
            )

        total_notes += 1

    print("\n===== PHI GUARDIAN MODEL EVALUATION =====")
    print("Dataset:", DATASET_PATH)
    print("Total notes evaluated:", total_notes)

    total_tp = total_fp = total_fn = 0

    print("\n===== PER-ENTITY RESULTS =====")

    for label in LABELS:
        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"\n{label}")
        print(f"  TP: {tp}")
        print(f"  FP: {fp}")
        print(f"  FN: {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\n===== OVERALL MICRO RESULTS =====")
    print("True Positives:", total_tp)
    print("False Positives:", total_fp)
    print("False Negatives:", total_fn)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n===== EVALUATION COMPLETED =====")
    print(
        "Note: These results evaluate the API inference pipeline, "
        "including its deterministic regex corrections."
    )


if __name__ == "__main__":
    main()
