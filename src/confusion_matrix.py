
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.deidentifier import find_entities


DATASET_PATH = Path("data/raw/synthetic_clinical_notes.json")


def generate_confusion_matrix():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    labels = ["PERSON", "PHONE", "EMAIL", "MRN"]

    y_true = []
    y_pred = []

    for record in dataset:
        expected = record["entities"]
        predicted = find_entities(record["text"])

        for label in labels:
            expected_values = {
                value.casefold()
                for key, value in expected.items()
                if key == label
            }

            predicted_values = {
                entity["text"].casefold()
                for entity in predicted
                if entity["label"] == label
            }

            y_true.extend([1] * len(expected_values))
            y_pred.extend([
                1 if value in predicted_values else 0
                for value in expected_values
            ])

            # Include false positive detections
            false_positives = predicted_values - expected_values

            y_true.extend([0] * len(false_positives))
            y_pred.extend([1] * len(false_positives))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not PHI", "PHI"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("PHI Detection Confusion Matrix")
    plt.tight_layout()

    output_path = Path("reports")
    output_path.mkdir(exist_ok=True)

    plt.savefig(
        output_path / "confusion_matrix.png",
        dpi=300
    )

    print("\nConfusion matrix saved to:")
    print("reports/confusion_matrix.png")

    plt.show()


if __name__ == "__main__":
    generate_confusion_matrix()
