"""Run the complete real-data pipeline and create dashboard predictions."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import json
import pandas as pd

from src.data.ingestion import load_appointments_csv, save_processed
from src.data.preprocessing import clean_data
from src.features.feature_engineering import engineer_features
from src.models.train import train_and_compare
from src.models import registry
from src.api.routes.predict import _classify_tier, _load_thresholds

RAW = PROJECT_ROOT / "data/raw/KaggleV2-May-2016.csv"
PROC = PROJECT_ROOT / "data/processed"


def main():
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. Run: python scripts/download_real_data.py"
        )

    valid, audit = load_appointments_csv(RAW, training=True)
    if not valid.empty:
        print(f"Validated rows: {len(valid):,}")
    else:
        from collections import Counter
        reasons = Counter(item.get("reason", "unknown") for item in audit.get("errors", []))
        raise ValueError(
            "Data validation rejected every row. "
            f"Audit summary: {dict(reasons)}. "
            f"See {PROC / 'ingestion_audit.json'} for row-level details."
        )
    save_processed(valid, PROC / "ingested.csv")
    (PROC / "ingestion_audit.json").write_text(json.dumps(audit, indent=2, default=str))

    cleaned = clean_data(valid)
    cleaned.to_csv(PROC / "cleaned.csv", index=False)

    features = engineer_features(cleaned, encode=True)
    features.to_csv(PROC / "features.csv", index=False)

    summary = train_and_compare(str(PROC / "features.csv"), str(PROJECT_ROOT / "src/models/config/model_params.yaml"), registry_dir=str(PROJECT_ROOT / "models_registry"))
    (PROC / "model_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Create a dashboard-ready historical scoring file using the production model.
    model, metadata = registry.load_production_model()
    X = features.drop(columns=["appointment_id", "no_show"], errors="ignore")
    for col in metadata["feature_columns"]:
        if col not in X.columns:
            X[col] = 0
    X = X[metadata["feature_columns"]]
    proba = model.predict_proba(X)[:, 1]
    thresholds = _load_thresholds()

    out = cleaned[["appointment_id", "patient_id", "appointment_date"]].copy()
    out["no_show_probability"] = proba
    out["risk_tier"] = [
        _classify_tier(float(p), thresholds) for p in proba
    ]
    if "doctor" in cleaned.columns:
        out["doctor"] = cleaned["doctor"]
    out["no_show"] = cleaned["no_show"].astype(int)
    out = out.sort_values("no_show_probability", ascending=False)
    out.to_csv(PROC / "predictions.csv", index=False)

    print("\nREAL DATA PIPELINE COMPLETE")
    print(f"Rows ingested: {len(valid):,}")
    print(f"Rows used for training: {len(features):,}")
    print(f"Selected model: {summary['selected']}")
    print(f"Test metrics: {summary['test_metrics']}")
    print(f"Predictions: {PROC / 'predictions.csv'}")


if __name__ == "__main__":
    main()
