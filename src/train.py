
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from src.labels import LABELS


# Configuration
MODEL_NAME = "distilbert/distilbert-base-uncased"

DATA_PATH = Path("data/processed/tokenized_dataset.json")
MODEL_OUTPUT = Path("models/trained")
REPORT_PATH = Path("reports")

EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
VALIDATION_RATIO = 0.2
SEED = 42

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


label2id = {
    label: index
    for index, label in enumerate(LABELS)
}

id2label = {
    index: label
    for label, index in label2id.items()
}


# Dataset
class PHIDataset(Dataset):

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    with DATA_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if len(records) < 2:
        raise ValueError("At least two records are required.")

    for index, record in enumerate(records):

        ids = record["input_ids"]
        masks = record["attention_mask"]
        labels = record["labels"]

        if not (len(ids) == len(masks) == len(labels)):
            raise ValueError(
                f"Sequence length mismatch in record {index}"
            )

        for label in labels:
            if label != -100 and not 0 <= label < len(LABELS):
                raise ValueError(
                    f"Invalid label ID {label} in record {index}"
                )

    random.Random(SEED).shuffle(records)

    validation_size = max(
        1, round(len(records) * VALIDATION_RATIO)
    )

    validation_records = records[:validation_size]
    training_records = records[validation_size:]

    print("Total records:", len(records))
    print("Training records:", len(training_records))
    print("Validation records:", len(validation_records))

    return training_records, validation_records


# Dynamic batch padding
def collate_fn(batch):

    max_length = max(
        len(record["input_ids"]) for record in batch
    )

    input_ids = []
    attention_masks = []
    labels = []

    for record in batch:

        padding_length = max_length - len(record["input_ids"])

        input_ids.append(
            record["input_ids"] + [0] * padding_length
        )

        attention_masks.append(
            record["attention_mask"] + [0] * padding_length
        )

        labels.append(
            record["labels"] + [-100] * padding_length
        )

    return {
        "input_ids": torch.tensor(
            input_ids, dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            attention_masks, dtype=torch.long
        ),
        "labels": torch.tensor(
            labels, dtype=torch.long
        ),
    }


# Evaluation
def evaluate(model, loader):

    model.eval()

    true_labels = []
    predicted_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            total_loss += outputs.loss.item()

            predictions = torch.argmax(
                outputs.logits, dim=-1
            )

            for prediction, target in zip(
                predictions.cpu().tolist(),
                batch["labels"].cpu().tolist(),
            ):

                true_sequence = []
                predicted_sequence = []

                for pred, actual in zip(prediction, target):

                    if actual == -100:
                        continue

                    true_sequence.append(id2label[actual])
                    predicted_sequence.append(id2label[pred])

                true_labels.append(true_sequence)
                predicted_labels.append(predicted_sequence)

    metrics = {
        "loss": total_loss / len(loader),
        "precision": precision_score(
            true_labels, predicted_labels
        ),
        "recall": recall_score(
            true_labels, predicted_labels
        ),
        "f1": f1_score(
            true_labels, predicted_labels
        ),
        "classification_report": classification_report(
            true_labels, predicted_labels
        ),
    }

    return metrics


# Main training
def main():

    print("\n========== DISTILBERT TRAINING ==========")

    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)
    print("Labels:", len(LABELS))

    training_records, validation_records = load_dataset()

    train_dataset = PHIDataset(training_records)
    validation_dataset = PHIDataset(validation_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history = []

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        metrics = evaluate(
            model, validation_loader
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": metrics["loss"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

        history.append(epoch_result)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {metrics['loss']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            model.save_pretrained(MODEL_OUTPUT)
            tokenizer.save_pretrained(MODEL_OUTPUT)

            print("Best model saved successfully!")

            with (
                REPORT_PATH / "classification_report.txt"
            ).open("w", encoding="utf-8") as file:
                file.write(metrics["classification_report"])

    with (
        REPORT_PATH / "training_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print("\n========== TRAINING COMPLETED ==========")
    print("Best validation F1:", round(best_f1, 4))
    print("Model saved:", MODEL_OUTPUT)
    print("Reports saved:", REPORT_PATH)
    print("=========================================")


if __name__ == "__main__":
    main()
