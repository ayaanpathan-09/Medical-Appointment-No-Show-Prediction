"""
POST /api/v1/predict/batch -- score many appointments at once (FR-PE-03),
used for nightly re-scoring of the schedule.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.auth import verify_api_key
from src.api.routes.predict import _classify_tier, _load_thresholds, _request_to_features
from src.api.schemas import BatchPredictRequest, BatchPredictResponse, PredictResponse
from src.models import registry

router = APIRouter()


@router.post("/predict/batch", response_model=BatchPredictResponse, dependencies=[Depends(verify_api_key)])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    model, metadata = registry.load_production_model()
    thresholds = _load_thresholds()
    feature_columns = metadata["feature_columns"]

    predictions: list[PredictResponse] = []
    for appt in request.appointments:
        X = _request_to_features(appt, feature_columns)
        probability = float(model.predict_proba(X)[0, 1])
        tier = _classify_tier(probability, thresholds)
        predictions.append(
            PredictResponse(
                appointment_id=appt.appointment_id,
                no_show_probability=round(probability, 4),
                risk_tier=tier,
                model_version=metadata["model_version"],
            )
        )

    return BatchPredictResponse(predictions=predictions)
