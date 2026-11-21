import json
from pathlib import Path

from transformers import AutoTokenizer

from src.labels import LABELS

# Project paths
INPUT_PATH = Path("data/raw/synthetic_clinical_notes.json")
OUTPUT_PATH = Path("data/processed/tokenized_dataset.json")

MODEL_NAME = "distilbert/distilbert-base-uncased"

# Map dataset annotation types to model label types
ENTITY_MAPPING = {
    "PERSON": "NAME",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "MRN": "MRN",
}

label2id = {
    label: index
    for index, label in enumerate(LABELS)
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def find_entity_spans(text, entities):
    """Find every occurrence of each annotated entity."""
    spans = []

    for entity_type, value in entities.items():
        mapped_type = ENTITY_MAPPING.get(entity_type)

        if mapped_type is None:
            continue

        start = 0

        while True:
            start = text.find(value, start)

            if start == -1:
                break

            end = start + len(value)
            spans.append((start, end, mapped_type))
            start = end

    return sorted(spans, key=lambda item: item[0])


def create_character_labels(text, spans):
    """Create character-level BIO labels."""
    labels = ["O"] * len(text)

    for start, end, entity_type in spans:
        labels[start] = f"B-{entity_type}"

        for position in range(start + 1, end):
            labels[position] = f"I-{entity_type}"

    return labels



def tokenize_and_align(text, character_labels):
    """Tokenize text and correctly align BIO labels."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    token_labels = []

    for start, end in encoding["offset_mapping"]:
        if start == end:
            token_labels.append(-100)
            continue

        token_label = character_labels[start]

        if token_label.startswith("I-"):
            entity_type = token_label[2:]
            previous_label = (
                character_labels[start - 1]
                if start > 0
                else "O"
            )

            # Keep I-label when the previous character
            # belongs to the same entity.
            if previous_label not in (
                f"B-{entity_type}",
                f"I-{entity_type}",
            ):
                token_label = f"B-{entity_type}"

        token_labels.append(label2id[token_label])

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": token_labels,
    }



def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    processed_records = []

    for index, record in enumerate(records, start=1):
        text = record["text"]
        entities = record.get("entities", {})

        spans = find_entity_spans(text, entities)

        character_labels = create_character_labels(
            text,
            spans,
        )

        processed = tokenize_and_align(
            text,
            character_labels,
        )

        processed_records.append({
            "text": text,
            "entities": entities,
            "template_id": record["template_id"],
            **processed,
        })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            processed_records,
            file,
            indent=2,
        )

    print("\n========== DATASET PREPROCESSING ==========")
    print("Original records:", len(records))
    print("Processed records:", len(processed_records))
    print("Total labels:", len(LABELS))
    print("Output file:", OUTPUT_PATH)
    print("===========================================")


if __name__ == "__main__":
    main()
