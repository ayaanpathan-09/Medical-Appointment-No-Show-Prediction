"""
Integration test for the Scoring API. Trains a small model on the sample
dataset on the fly so the test does not depend on a pre-existing
production model (TRD Section 9: Integration Tests).
"""
import os

import pytest
from fastapi.testclient import TestClient

from src.data.ingestion import load_appointments_csv
from src.data.preprocessing import clean_data
from src.features.feature_engineering import engineer_features
from src.models.train import train_and_compare

os.environ.setdefault("API_KEYS", "test-key")


@pytest.fixture(scope="module")
def trained_registry(tmp_path_factory):
    registry_dir = tmp_path_factory.mktemp("models_registry")
    valid_df, _ = load_appointments_csv("tests/test_data/sample_appointments.csv", training=True)
    cleaned = clean_data(valid_df)
    features = engineer_features(cleaned)
    features_path = tmp_path_factory.mktemp("data") / "features.csv"
    features.to_csv(features_path, index=False)
    train_and_compare(str(features_path), "src/models/config/model_params.yaml", registry_dir=str(registry_dir))
    return registry_dir


def test_predict_endpoint_requires_api_key(trained_registry, monkeypatch):
    from src.models import registry
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_DIR", trained_registry)
    from src.api.main import app
    client = TestClient(app)

    resp = client.post("/api/v1/predict", json={})
    assert resp.status_code in (401, 422)


def test_predict_endpoint_returns_valid_response(trained_registry, monkeypatch):
    from src.models import registry
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_DIR", trained_registry)
    from src.api.main import app
    client = TestClient(app)

    payload = {
        "appointment_id": "T1",
        "patient_id": "P100",
        "age": 40,
        "gender": "F",
        "scheduled_date": "2024-01-01",
        "appointment_date": "2024-01-10",
        "sms_received": True,
        "chronic_condition": False,
    }
    resp = client.post("/api/v1/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["no_show_probability"] <= 1.0
    assert body["risk_tier"] in ("low", "medium", "high")
