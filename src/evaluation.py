
import json
from collections import defaultdict
from pathlib import Path

from src.deidentifier import find_entities


DATASET_PATH = Path("data/raw/synthetic_clinical_notes.json")

PHI_LABELS = ["PERSON", "PHONE", "EMAIL", "MRN"]


def evaluate_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    metrics = defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0}
    )

    for record in dataset:
        text = record["text"]
        expected_entities = record["entities"]

        predicted_entities = find_entities(text)

        for label in PHI_LABELS:
            expected = {
                value.casefold()
                for entity_label, value in expected_entities.items()
                if entity_label == label
            }

            predicted = {
                entity["text"].casefold()
                for entity in predicted_entities
                if entity["label"] == label
            }

            metrics[label]["TP"] += len(expected & predicted)
            metrics[label]["FP"] += len(predicted - expected)
            metrics[label]["FN"] += len(expected - predicted)

    print("\n========== PHI EVALUATION REPORT ==========")
    print("Total clinical notes:", len(dataset))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in PHI_LABELS:
        tp = metrics[label]["TP"]
        fp = metrics[label]["FP"]
        fn = metrics[label]["FN"]

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"\n{label}")
        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

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

    print("\n========== OVERALL MICRO AVERAGE ==========")
    print("Total TP:", total_tp)
    print("Total FP:", total_fp)
    print("Total FN:", total_fn)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


if __name__ == "__main__":
    evaluate_dataset()
