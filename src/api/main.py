"""
Scoring API entry point (TRD Section 5).

Run with:
    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import batch_predict, predict

app = FastAPI(
    title="Medical Appointment No-Show Prediction API",
    version="1.0",
    description="Scoring API for the Medical Appointment No-Show Prediction System.",
)

app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])
app.include_router(batch_predict.router, prefix="/api/v1", tags=["prediction"])


@app.get("/health", tags=["system"])
def health_check() -> dict:
    return {"status": "ok"}
