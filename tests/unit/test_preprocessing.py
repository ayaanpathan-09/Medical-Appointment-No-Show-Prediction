import pandas as pd

from src.data.preprocessing import clean_data


def _sample_df():
    return pd.DataFrame({
        "appointment_id": ["A1", "A1", "A2", "A3"],
        "patient_id": ["P1", "P1", "P1", "P2"],
        "gender": ["F", "F", "F", None],
        "age": [30, 30, 200, 45],
        "scheduled_date": ["2024-01-01", "2024-01-01", "2024-01-05", "2024-01-10"],
        "appointment_date": ["2024-01-10", "2024-01-10", "2024-01-01", "2024-01-12"],
        "sms_received": [1, 1, 0, None],
        "chronic_condition": [0, 0, 1, None],
        "no_show": [0, 0, 1, 0],
    })


def test_clean_data_deduplicates_on_appointment_id():
    cleaned = clean_data(_sample_df())
    assert cleaned["appointment_id"].is_unique


def test_clean_data_caps_age_and_flags_negative_waiting_days():
    cleaned = clean_data(_sample_df())
    assert cleaned["age"].max() <= 120
    # A2 has appointment_date before scheduled_date -> negative waiting days flagged
    flagged = cleaned.loc[cleaned["appointment_id"] == "A2", "waiting_days_raw_flag"].iloc[0]
    assert flagged == 1


def test_clean_data_imputes_missing_values():
    cleaned = clean_data(_sample_df())
    assert cleaned["gender"].isna().sum() == 0
    assert cleaned["sms_received"].isna().sum() == 0
