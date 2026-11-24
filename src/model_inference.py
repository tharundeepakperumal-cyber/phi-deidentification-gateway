
import os
import re
import threading
from pathlib import Path

import torch

torch.set_num_threads(1)
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "template_aware"
)

HF_MODEL_ID = os.getenv("HF_MODEL_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_LENGTH = 512
STRIDE = 64

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

_tokenizer = None

_model = None

_model_lock = threading.Lock()

_inference_lock = threading.Lock()

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
def _is_mrn_field_label_prediction(text, entity):
    """Reject model predictions overlapping an MRN field label."""

    entity_start = entity["start"]
    entity_end = entity["end"]

    for match in re.finditer(r"\bMRN\s*:", text, re.IGNORECASE):
        overlaps_label = (
            entity_start < match.end()
            and entity_end > match.start()
        )

        if overlaps_label:
            return True

    return False



def load_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    with _model_lock:
        if _model is None:
            if HF_MODEL_ID:
                model_source = HF_MODEL_ID
                model_token = HF_TOKEN
            else:
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"Model directory not found: {MODEL_PATH}"
                    )
                model_source = str(MODEL_PATH)
                model_token = None

            _tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                use_fast=True,
                token=model_token,
            )

            _model = AutoModelForTokenClassification.from_pretrained(
                model_source,
                token=model_token,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

            _model.to(DEVICE)
            _model.eval()

    return _tokenizer, _model


def resolve_overlapping_entities(entities):
    """Resolve overlapping entities using priority and span length."""

    entities = sorted(
        entities,
        key=lambda item: (
            -item["priority"],
            -(item["end"] - item["start"]),
            item["start"],
        ),
    )

    resolved = []

    for entity in entities:
        overlaps = any(
            entity["start"] < existing["end"]
            and entity["end"] > existing["start"]
            for existing in resolved
        )

        if not overlaps:
            resolved.append(entity)

    return sorted(resolved, key=lambda item: item["start"])

def extract_entities(text):
    with _inference_lock:
        return _extract_entities_unlocked(text)


def _extract_entities_unlocked(text):
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

    chunk_count = encoding["input_ids"].shape[0]
    entities = []

    for chunk_index in range(chunk_count):
        chunk_inputs = {
            key: value[chunk_index:chunk_index + 1].to(DEVICE)
            for key, value in encoding.items()
        }

        chunk_offsets = offsets[chunk_index].tolist()

        with torch.inference_mode():
            outputs = model(**chunk_inputs)
            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )[0].cpu().tolist()

        del outputs
        del chunk_inputs

        current = None

        def flush():
            nonlocal current
            if current is not None:
                entities.append(current)
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

    # Remove model predictions that identify an MRN field label.
    entities = [
        entity
        for entity in entities
        if not _is_mrn_field_label_prediction(text, entity)
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

    # Resolve overlapping model and regex predictions.
    return resolve_overlapping_entities(entities)



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
