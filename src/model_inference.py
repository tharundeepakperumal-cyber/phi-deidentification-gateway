
import re
import threading
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "template_aware"
)

MAX_LENGTH = 512
STRIDE = 64

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

_tokenizer = None
_model = None
_model_lock = threading.Lock()

CORRECTION_PATTERNS = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE": re.compile(
        r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?"
        r"(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
    ),
    "MRN": re.compile(
        r"\bMRN[-:\s]*\d{5,12}\b",
        re.IGNORECASE,
    ),
}


def load_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    with _model_lock:
        if _model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {MODEL_PATH}"
                )

            _tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_PATH),
                use_fast=True,
            )

            _model = AutoModelForTokenClassification.from_pretrained(
                str(MODEL_PATH)
            )

            _model.to(DEVICE)
            _model.eval()

    return _tokenizer, _model


def extract_entities(text):
    tokenizer, model = load_model()

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

    inputs = {
        key: value.to(DEVICE)
        for key, value in encoding.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    predictions = torch.argmax(
        outputs.logits,
        dim=-1,
    ).cpu().tolist()

    entities = []

    for chunk_predictions, chunk_offsets in zip(
        predictions,
        offsets.tolist(),
    ):
        current = None

        def flush():
            nonlocal current

            if current is not None:
                entities.append(current)
                current = None

        for prediction, (start, end) in zip(
            chunk_predictions,
            chunk_offsets,
        ):
            if start == end:
                flush()
                continue

            label = model.config.id2label[prediction]

            if label == "O":
                flush()
                continue

            if "-" not in label:
                flush()
                continue

            prefix, entity_type = label.split("-", 1)

            if prefix == "B":
                flush()
                current = {
                    "start": start,
                    "end": end,
                    "label": entity_type,
                    "text": text[start:end],
                    "priority": 1,
                }

            elif prefix == "I":
                if (
                    current is None
                    or current["label"] != entity_type
                ):
                    flush()
                    current = {
                        "start": start,
                        "end": end,
                        "label": entity_type,
                        "text": text[start:end],
                        "priority": 1,
                    }
                else:
                    current["end"] = end
                    current["text"] = text[
                        current["start"]:end
                    ]

            else:
                flush()

        flush()


    # Remove invalid EMAIL predictions from the model.
    entities = [
        entity
        for entity in entities
        if (
            entity["label"] != "EMAIL"
            or CORRECTION_PATTERNS["EMAIL"].fullmatch(
                entity["text"]
            )
        )
    ]

    # Add regex-based corrections for common identifiers.
    for entity_type, pattern in CORRECTION_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({
                "start": match.start(),
                "end": match.end(),
                "label": entity_type,
                "text": match.group(),
                "priority": 2,
            })


    # Prefer regex matches when overlapping model predictions occur.
    entities.sort(
        key=lambda item: (
            item["start"],
            -item["priority"],
            -(item["end"] - item["start"]),
        )
    )

    resolved = []

    for entity in entities:
        if not resolved or entity["start"] >= resolved[-1]["end"]:
            resolved.append(entity)
        elif entity["priority"] > resolved[-1]["priority"]:
            resolved[-1] = entity

    return resolved


def deidentify_with_model(text):
    entities = extract_entities(text)

    placeholder_lookup = {}
    original_values = {}
    counters = {}
    replacements = []

    for entity in entities:
        label = entity["label"]
        original = entity["text"]

        # Case-insensitive lookup while preserving original text.
        lookup_key = (label, original.casefold())

        if lookup_key not in placeholder_lookup:
            counters[label] = counters.get(label, 0) + 1

            placeholder = f"[{label}_{counters[label]}]"

            placeholder_lookup[lookup_key] = placeholder
            original_values[lookup_key] = original

        replacements.append((
            entity["start"],
            entity["end"],
            placeholder_lookup[lookup_key],
        ))

    masked_text = text

    for start, end, placeholder in sorted(
        replacements,
        key=lambda item: item[0],
        reverse=True,
    ):
        masked_text = (
            masked_text[:start]
            + placeholder
            + masked_text[end:]
        )

    output_mapping = {
        placeholder_lookup[key]: original_values[key]
        for key in placeholder_lookup
    }

    return masked_text, output_mapping
