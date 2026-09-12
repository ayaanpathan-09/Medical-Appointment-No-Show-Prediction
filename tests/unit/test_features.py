import pandas as pd

from src.features.feature_engineering import engineer_features


def _sample_df():
    return pd.DataFrame({
        "appointment_id": ["A1", "A2", "A3"],
        "patient_id": ["P1", "P1", "P2"],
        "gender": ["F", "F", "M"],
        "age": [8, 25, 65],
        "scheduled_date": ["2024-01-01", "2024-02-01", "2024-01-01"],
        "appointment_date": ["2024-01-05", "2024-02-10", "2024-01-03"],
        "sms_received": [1, 0, 1],
        "chronic_condition": [0, 0, 1],
        "no_show": [1, 0, 0],
    })


def test_engineer_features_produces_expected_row_count():
    features = engineer_features(_sample_df())
    assert len(features) == 3
    assert "no_show" in features.columns


def test_prior_noshow_rate_uses_only_past_appointments():
    features = engineer_features(_sample_df(), encode=False)
    # P1's 2nd appointment (A2) should reflect P1's 1st appointment (A1, no_show=1)
    rate_a2 = features.loc[features["appointment_id"] == "A2", "prior_noshow_rate"].iloc[0]
    assert rate_a2 == 1.0
    # P1's 1st appointment (A1) has no prior history -> rate 0
    rate_a1 = features.loc[features["appointment_id"] == "A1", "prior_noshow_rate"].iloc[0]
    assert rate_a1 == 0.0


def test_age_group_binning():
    features = engineer_features(_sample_df(), encode=False)
    assert set(features["age_group"]) <= {"0-12", "13-19", "20-40", "41-60", "60+"}
