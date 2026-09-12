"""
POST /api/v1/predict -- score a single upcoming appointment
(FR-PE-01, FR-PE-02, FR-PE-05).
"""
from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path
from fastapi import APIRouter, Depends

from src.api.auth import verify_api_key
from src.api.schemas import PredictRequest, PredictResponse
from src.features.feature_engineering import engineer_features
from src.models import registry

router = APIRouter()


def _load_thresholds() -> dict:
    """Risk tier thresholds (FRD Section 7, configurable by administrators)."""
    project_root = Path(__file__).resolve().parents[3]
    with open(project_root / "config" / "thresholds.yaml") as f:
        return yaml.safe_load(f)


def _classify_tier(probability: float, thresholds: dict) -> str:
    if probability >= thresholds["high"]:
        return "high"
    if probability >= thresholds["medium"]:
        return "medium"
    return "low"


def _request_to_features(req: PredictRequest, feature_columns: list[str]) -> pd.DataFrame:
    row = {
        "appointment_id": req.appointment_id,
        "patient_id": req.patient_id,
        "age": req.age,
        "gender": req.gender,
        "scheduled_date": req.scheduled_date,
        "appointment_date": req.appointment_date,
        "sms_received": int(req.sms_received),
        "chronic_condition": int(req.chronic_condition),
        "appointment_type": req.appointment_type or "Unknown",
    }
    df = pd.DataFrame([row])
    features = engineer_features(df, encode=True)
    if req.prior_noshow_rate is not None:
        features["prior_noshow_rate"] = req.prior_noshow_rate

    # Align to the exact columns the production model was trained on
    # (one-hot columns for a single row won't include every category).
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0
    features = features[feature_columns]
    return features


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
def predict(request: PredictRequest) -> PredictResponse:
    model, metadata = registry.load_production_model()
    thresholds = _load_thresholds()

    X = _request_to_features(request, metadata["feature_columns"])
    probability = float(model.predict_proba(X)[0, 1])
    tier = _classify_tier(probability, thresholds)

    return PredictResponse(
        appointment_id=request.appointment_id,
        no_show_probability=round(probability, 4),
        risk_tier=tier,
        model_version=metadata["model_version"],
    )
