from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
CLEANED_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "predictions.csv"


def classify_risk(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    elif probability >= 0.40:
        return "medium"
    return "low"


def main():
    print("Loading production model...")

    model, metadata = registry.load_production_model()

    print(f"Model version: {metadata['model_version']}")
    print(f"Model: {metadata.get('algorithm', 'random_forest')}")

    features_df = pd.read_csv(FEATURES_FILE)
    cleaned_df = pd.read_csv(CLEANED_FILE)

    feature_columns = metadata["feature_columns"]

    # Prepare model input
    X = features_df.drop(
        columns=["appointment_id", "no_show"],
        errors="ignore",
    ).copy()

    # Make sure columns match the trained model exactly
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    print(f"Scoring {len(X)} appointments...")

    probabilities = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame({
        "appointment_id": features_df["appointment_id"],
        "no_show_probability": probabilities,
    })

    predictions["risk_tier"] = predictions["no_show_probability"].apply(
        classify_risk
    )

    # Add patient/date information when available
    useful_columns = [
        "appointment_id",
        "patient_id",
        "appointment_date",
        "doctor",
        "no_show",
    ]

    available_columns = [
        col for col in useful_columns
        if col in cleaned_df.columns
    ]

    if "appointment_id" in cleaned_df.columns:
        extra = cleaned_df[available_columns].drop_duplicates(
            subset=["appointment_id"]
        )

        predictions = predictions.merge(
            extra,
            on="appointment_id",
            how="left",
            suffixes=("", "_source"),
        )

    # Keep dashboard-friendly column order
    preferred_order = [
        "appointment_id",
        "patient_id",
        "appointment_date",
        "no_show_probability",
        "risk_tier",
        "doctor",
        "no_show",
    ]

    final_columns = [
        col for col in preferred_order
        if col in predictions.columns
    ]

    predictions = predictions[final_columns]

    predictions["no_show_probability"] = predictions[
        "no_show_probability"
    ].round(4)

    predictions.to_csv(OUTPUT_FILE, index=False)

    print()
    print("SUCCESS!")
    print(f"Predictions saved to:")
    print(OUTPUT_FILE)
    print()
    print(f"Total predictions: {len(predictions)}")
    print()
    print("Risk distribution:")
    print(predictions["risk_tier"].value_counts())
    print()
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()