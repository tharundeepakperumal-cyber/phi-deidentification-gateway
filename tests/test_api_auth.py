
from fastapi.testclient import TestClient

import src.api as api


client = TestClient(api.app)


def test_deidentify_without_api_key(monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "test-secret-key")

    response = client.post(
        "/deidentify",
        json={"text": "Patient John Smith"}
    )

    assert response.status_code == 401


def test_deidentify_with_invalid_api_key(monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "test-secret-key")

    response = client.post(
        "/deidentify",
        headers={"X-API-Key": "wrong-key"},
        json={"text": "Patient John Smith"}
    )

    assert response.status_code == 401


def test_deidentify_with_valid_api_key(monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "test-secret-key")

    response = client.post(
        "/deidentify",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "text": "Patient John Smith, email john@example.com"
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert "John Smith" not in result["deidentifier_text"]
    assert "john@example.com" not in result["deidentifier_text"]
    assert result["detected_entities"]["PERSON"] >= 1
    assert result["detected_entities"]["EMAIL"] >= 1


def test_api_key_not_configured(monkeypatch):
    monkeypatch.setattr(api, "API_KEY", None)

    response = client.post(
        "/deidentify",
        json={"text": "Patient John Smith"}
    )

    assert response.status_code == 503
