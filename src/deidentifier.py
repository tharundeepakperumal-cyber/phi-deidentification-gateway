import re
from typing import Dict, List, Tuple


PATTERNS = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    "PHONE": re.compile(
        r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?"
        r"(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
    ),

    "MRN": re.compile(
        r"\bMRN[-:\s]*\d{5,12}\b",
        re.IGNORECASE
    ),

    "PERSON": re.compile(
        r"\b(?:Patient\s+)?"
        r"([A-Z][a-z]+(?:['’-][A-Z]?[a-z]+)?"
        r"(?:\s+[A-Z][a-z]+(?:['’-][A-Z]?[a-z]+)?){1,3})\b"
    )
}


def find_entities(text: str) -> List[dict]:
    """Find supported PHI identifiers."""

    entities = []

    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):

            if label == "PERSON":
                start, end = match.span(1)
                matched_text = match.group(1)
            else:
                start, end = match.span()
                matched_text = match.group()

            entities.append({
                "start": start,
                "end": end,
                "text": matched_text,
                "label": label
            })

    entities.sort(
        key=lambda x: (
            x["start"],
            -(x["end"] - x["start"])
        )
    )

    resolved = []

    for entity in entities:
        if not resolved or entity["start"] >= resolved[-1]["end"]:
            resolved.append(entity)

    return resolved


def deidentify(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace detected PHI with consistent placeholders."""

    entities = find_entities(text)

    mapping = {}
    placeholder_lookup = {}
    counters = {}
    replacements = []

    for entity in entities:
        original = entity["text"]
        label = entity["label"]

        key = (label, original.casefold())

        if key not in placeholder_lookup:
            counters[label] = counters.get(label, 0) + 1

            placeholder = f"[{label}_{counters[label]}]"

            placeholder_lookup[key] = placeholder
            mapping[placeholder] = original
        else:
            placeholder = placeholder_lookup[key]

        replacements.append({
            "start": entity["start"],
            "end": entity["end"],
            "placeholder": placeholder
        })

    masked_text = text

    for item in sorted(
        replacements,
        key=lambda x: x["start"],
        reverse=True
    ):
        masked_text = (
            masked_text[:item["start"]]
            + item["placeholder"]
            + masked_text[item["end"]:]
        )

    return masked_text, mapping
