
from src.rehydrator import rehydrate


def test_restore_known_placeholders():
    mapping = {
        "[PERSON_1]": "John Smith",
        "[PHONE_1]": "9876543210",
        "[EMAIL_1]": "john@example.com"
    }

    response = "Patient [PERSON_1], Phone [PHONE_1], Email [EMAIL_1]"

    result = rehydrate(response, mapping)

    assert result == (
        "Patient John Smith, Phone 9876543210, "
        "Email john@example.com"
    )


def test_unknown_placeholder_remains_unchanged():
    response = "Patient [PERSON_1] and [PERSON_2]"

    mapping = {
        "[PERSON_1]": "John Smith"
    }

    result = rehydrate(response, mapping)

    assert result == "Patient John Smith and [PERSON_2]"


def test_empty_mapping():
    response = "Patient [PERSON_1]"

    result = rehydrate(response, {})

    assert result == response


def test_overlapping_placeholders():
    mapping = {
        "[PERSON_1]": "John",
        "[PERSON_10]": "David"
    }

    response = "[PERSON_10] and [PERSON_1]"

    result = rehydrate(response, mapping)

    assert result == "David and John"
