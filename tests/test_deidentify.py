
import unittest

from src.deidentifier import deidentify, find_entities


class TestDeidentifier(unittest.TestCase):

    def test_person_detection(self):
        text = "Patient Aryan Maharaja visited hospital."
        masked, mapping = deidentify(text)

        self.assertIn("[PERSON_1]", masked)
        self.assertIn("Aryan Maharaja", mapping.values())

    def test_apostrophe_name_detection(self):
        text = "Patient Elisa Dalai visited hospital."
        masked, mapping = deidentify(text)

        self.assertIn("[PERSON_1]", masked)
        self.assertIn("Elisa Dalai", mapping.values())

    def test_phone_detection(self):
        text = "Contact: +919876543210"
        masked, mapping = deidentify(text)

        self.assertIn("[PHONE_1]", masked)
        self.assertIn("+919876543210", mapping.values())

    def test_email_detection(self):
        text = "Email: tharun@example.com"
        masked, mapping = deidentify(text)

        self.assertIn("[EMAIL_1]", masked)
        self.assertIn("tharun@example.com", mapping.values())

    def test_mrn_detection(self):
        text = "Medical record number: MRN123456"
        masked, mapping = deidentify(text)

        self.assertIn("[MRN_1]", masked)
        self.assertIn("MRN123456", mapping.values())

    def test_repeated_person_detection(self):
        text = "Aryan Maharaja visited. Aryan Maharaja returned."
        masked, mapping = deidentify(text)

        self.assertEqual(masked.count("[PERSON_1]"), 2)
        self.assertEqual(len(mapping), 1)

    def test_no_phi_text(self):
        text = "The patient needs a follow-up appointment."
        masked, mapping = deidentify(text)

        self.assertEqual(masked, text)
        self.assertEqual(mapping, {})


if __name__ == "__main__":
    unittest.main()
