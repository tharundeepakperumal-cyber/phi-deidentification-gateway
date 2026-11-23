import unittest
from unittest.mock import patch

from src.model_inference import (
    deidentify_with_model,
    resolve_overlapping_entities,
)

class TestEntityResolution(unittest.TestCase):

    def test_regex_entity_overrides_model_prediction(self):
        entities = [
            {
                "start": 10,
                "end": 30,
                "label": "NAME",
                "text": "person@example.com",
                "priority": 1,
            },
            {
                "start": 10,
                "end": 30,
                "label": "EMAIL",
                "text": "person@example.com",
                "priority": 2,
            },
        ]

        resolved = resolve_overlapping_entities(entities)

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["label"], "EMAIL")

    def test_longer_entity_wins_equal_priority(self):
        entities = [
            {
                "start": 5,
                "end": 12,
                "label": "NAME",
                "text": "Aryan M",
                "priority": 1,
            },
            {
                "start": 5,
                "end": 19,
                "label": "NAME",
                "text": "Aryan Maharaja",
                "priority": 1,
            },
        ]

        resolved = resolve_overlapping_entities(entities)

        self.assertEqual(len(resolved), 1)
        self.assertEqual(
            resolved[0]["text"],
            "Aryan Maharaja",
        )

    def test_non_overlapping_entities_are_preserved(self):
        entities = [
            {
                "start": 20,
                "end": 30,
                "label": "PHONE",
                "text": "9876543210",
                "priority": 2,
            },
            {
                "start": 0,
                "end": 5,
                "label": "NAME",
                "text": "Aryan",
                "priority": 1,
            },
        ]

        resolved = resolve_overlapping_entities(entities)

        self.assertEqual(len(resolved), 2)
        self.assertEqual(
            [entity["label"] for entity in resolved],
            ["NAME", "PHONE"],
        )


class TestModelDeidentification(unittest.TestCase):

    def test_person_replacement(self):
        text = "Patient Aryan Maharaja visited hospital."

        entities = [{
            "start": 8,
            "end": 22,
            "label": "NAME",
            "text": "Aryan Maharaja",
            "priority": 1,
        }]

        with patch(
            "src.model_inference.extract_entities",
            return_value=entities,
        ):
            masked, mapping = deidentify_with_model(text)

        self.assertIn("[NAME_1]", masked)
        self.assertEqual(mapping["[NAME_1]"], "Aryan Maharaja")

    def test_repeated_person_replacement(self):
        text = "Aryan Maharaja visited. Aryan Maharaja returned."

        first = text.index("Aryan Maharaja")
        second = text.index("Aryan Maharaja", first + 1)

        entities = [
            {
                "start": first,
                "end": first + len("Aryan Maharaja"),
                "label": "NAME",
                "text": "Aryan Maharaja",
                "priority": 1,
            },
            {
                "start": second,
                "end": second + len("Aryan Maharaja"),
                "label": "NAME",
                "text": "Aryan Maharaja",
                "priority": 1,
            },
        ]

        with patch(
            "src.model_inference.extract_entities",
            return_value=entities,
        ):
            masked, mapping = deidentify_with_model(text)

        self.assertEqual(masked.count("[NAME_1]"), 2)
        self.assertEqual(len(mapping), 1)

    def test_multiple_entity_types(self):
        text = "Patient Aryan Maharaja Email: aryan@example.com"

        name = "Aryan Maharaja"
        email = "aryan@example.com"

        entities = [
            {
                "start": text.index(name),
                "end": text.index(name) + len(name),
                "label": "NAME",
                "text": name,
                "priority": 1,
            },
            {
                "start": text.index(email),
                "end": text.index(email) + len(email),
                "label": "EMAIL",
                "text": email,
                "priority": 2,
            },
        ]

        with patch(
            "src.model_inference.extract_entities",
            return_value=entities,
        ):
            masked, mapping = deidentify_with_model(text)

        self.assertIn("[NAME_1]", masked)
        self.assertIn("[EMAIL_1]", masked)
        self.assertEqual(mapping["[NAME_1]"], name)
        self.assertEqual(mapping["[EMAIL_1]"], email)

    def test_case_insensitive_repeated_names(self):
        text = "Aryan Maharaja visited. ARYAN MAHARAJA returned."

        first = text.index("Aryan Maharaja")
        second = text.index("ARYAN MAHARAJA")

        entities = [
            {
                "start": first,
                "end": first + len("Aryan Maharaja"),
                "label": "NAME",
                "text": "Aryan Maharaja",
                "priority": 1,
            },
            {
                "start": second,
                "end": second + len("ARYAN MAHARAJA"),
                "label": "NAME",
                "text": "ARYAN MAHARAJA",
                "priority": 1,
            },
        ]

        with patch(
            "src.model_inference.extract_entities",
            return_value=entities,
        ):
            masked, mapping = deidentify_with_model(text)

        self.assertEqual(masked.count("[NAME_1]"), 2)
        self.assertEqual(len(mapping), 1)

    def test_no_phi_text(self):
        text = "The patient needs a follow-up appointment."

        with patch(
            "src.model_inference.extract_entities",
            return_value=[],
        ):
            masked, mapping = deidentify_with_model(text)

        self.assertEqual(masked, text)
        self.assertEqual(mapping, {})

    def test_multiple_entities_preserve_text_order(self):
        text = "Name: Aryan. Contact: 9876543210. MRN: MRN123456"

        entities = []

        for label, value in [
            ("NAME", "Aryan"),
            ("PHONE", "9876543210"),
            ("MRN", "MRN123456"),
        ]:
            start = text.index(value)
            entities.append({
                "start": start,
                "end": start + len(value),
                "label": label,
                "text": value,
                "priority": 1,
            })

        with patch(
            "src.model_inference.extract_entities",
            return_value=entities,
        ):
            masked, mapping = deidentify_with_model(text)

        self.assertLess(
            masked.index("[NAME_1]"),
            masked.index("[PHONE_1]"),
        )
        self.assertLess(
            masked.index("[PHONE_1]"),
            masked.index("[MRN_1]"),
        )
        self.assertEqual(len(mapping), 3)


if __name__ == "__main__":
    unittest.main()
