import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

MODEL_NAME = "models/template_aware"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("\n========== DISTILBERT INFERENCE TEST ==========")
print("Device:", DEVICE)

# Load trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

def extract_entities(text):

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    offsets = encoding.pop("offset_mapping")[0].tolist()

    inputs = {
        key: value.to(DEVICE)
        for key, value in encoding.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )[0].cpu().tolist()

    entities = []
    current_entity = None

    for prediction, (start, end) in zip(
        predictions, offsets
    ):

        if start == end:
            continue

        label = model.config.id2label[prediction]

        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B" or (
            current_entity
            and current_entity["label"] != entity_type
        ):

            if current_entity:
                entities.append(current_entity)

            current_entity = {
                "start": start,
                "end": end,
                "label": entity_type,
            }

        elif prefix == "I":

            if current_entity is None:
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": entity_type,
                }
            else:
                current_entity["end"] = end

    if current_entity:
        entities.append(current_entity)

       # Correct EMAIL and MRN boundaries using strict patterns.
    # Keep NAME and PHONE predictions from DistilBERT.

    entities = [
        entity for entity in entities
        if entity["label"] not in ("EMAIL", "MRN")
    ]

    correction_patterns = {
        "EMAIL": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "MRN": re.compile(
            r"\bMRN[- ]?\d{5,12}\b",
            re.IGNORECASE
        ),
    }

    for label, pattern in correction_patterns.items():
        for match in pattern.finditer(text):
            entities.append({
                "start": match.start(),
                "end": match.end(),
                "label": label,
            })

    entities.sort(key=lambda entity: entity["start"])

    results = []

    for entity in entities:
        results.append({
            "text": text[
                entity["start"]:entity["end"]
            ],
            "label": entity["label"],
            "start": entity["start"],
            "end": entity["end"],
        })

    return results


# New test examples
test_notes = [
    (
        "Patient Michael Anderson visited the clinic. "
        "Contact: +919876543210. "
        "Email: michael.anderson@hospital.org. "
        "Medical record number: MRN882341."
    ),
    (
        "Patient Priya Sharma was admitted yesterday. "
        "Her phone number is +919876512345. "
        "Email: priya.sharma@example.com. "
        "MRN: MRN765432."
    ),
    (
        "Patient Daniel Wilson requires a follow-up. "
        "Contact number: 9876543210. "
        "Email: daniel.wilson@example.org. "
        "Record ID: MRN123456."
    ),
]


for index, note in enumerate(test_notes, start=1):

    print(f"\nTest {index}")
    print("-" * 50)

    print("Input:", note)

    entities = extract_entities(note)

    print("\nDetected entities:")

    if not entities:
        print("No entities detected.")

    for entity in entities:
        print(
            f"{entity['label']}: {entity['text']}"
        )


print("\n========== INFERENCE TEST COMPLETED ==========")
print("\n========== EMAIL TOKEN DIAGNOSTICS ==========")

debug_text = (
    "Patient Michael Anderson visited the clinic. "
    "Contact: +919876543210. "
    "Email: michael.anderson@hospital.org. "
    "Medical record number: MRN882341."
)

encoding = tokenizer(
    debug_text,
    return_offsets_mapping=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

offsets = encoding.pop("offset_mapping")[0].tolist()

inputs = {
    key: value.to(DEVICE)
    for key, value in encoding.items()
}

with torch.inference_mode():
    outputs = model(**inputs)

predictions = torch.argmax(
    outputs.logits,
    dim=-1
)[0].cpu().tolist()

tokens = tokenizer.convert_ids_to_tokens(
    encoding["input_ids"][0].tolist()
)

for token, prediction, (start, end) in zip(
    tokens, predictions, offsets
):
    if start != end and (
        "michael.anderson@hospital.org".find(
            debug_text[start:end]
        ) >= 0
    ):
        print(
            f"Token: {token:15} "
            f"Label: {model.config.id2label[prediction]:12} "
            f"Text: {debug_text[start:end]!r}"
        )

print("\n========== DIAGNOSTICS COMPLETED ==========")