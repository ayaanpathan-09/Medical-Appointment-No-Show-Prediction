"""
Pydantic request/response models for the Scoring API (TRD Section 5).
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    appointment_id: str
    patient_id: str
    age: int = Field(ge=0, le=120)
    gender: Literal["M", "F"]
    scheduled_date: date
    appointment_date: date
    sms_received: bool
    chronic_condition: bool = False
    appointment_type: Optional[str] = None
    prior_noshow_rate: Optional[float] = Field(
        default=None,
        description="Optional pre-computed value; if omitted the API assumes 0.0 (new patient).",
    )


class PredictResponse(BaseModel):
    appointment_id: str
    no_show_probability: float
    risk_tier: Literal["low", "medium", "high"]
    model_version: str


class BatchPredictRequest(BaseModel):
    appointments: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
