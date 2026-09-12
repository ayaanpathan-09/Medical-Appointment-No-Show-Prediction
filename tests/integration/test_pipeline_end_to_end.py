"""
End-to-end pipeline test: ingestion -> preprocessing -> feature engineering
-> training, using the small sample dataset (TRD Section 9: Testing Strategy).
"""
from pathlib import Path

from src.data.ingestion import load_appointments_csv
from src.data.preprocessing import clean_data
from src.features.feature_engineering import engineer_features
from src.models.train import train_and_compare

SAMPLE_CSV = Path("tests/test_data/sample_appointments.csv")


def test_pipeline_end_to_end(tmp_path):
    valid_df, audit = load_appointments_csv(SAMPLE_CSV, training=True)
    assert audit["valid_record_count"] > 0

    cleaned = clean_data(valid_df)
    features = engineer_features(cleaned)
    assert "no_show" in features.columns

    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)

    summary = train_and_compare(
        str(features_path),
        "src/models/config/model_params.yaml",
        registry_dir=str(tmp_path / "models_registry"),
    )
    assert summary["selected"] in summary["candidates"]
    assert 0.0 <= summary["test_metrics"]["f1_score"] <= 1.0
