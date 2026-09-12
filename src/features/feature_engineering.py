"""
Feature engineering module.

Derives the modeling features listed in the TRD (Section 3.3):
waiting_days, appointment_dow, appointment_month, age_group,
prior_noshow_rate, sms_received, chronic_condition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

AGE_BINS = [-1, 12, 19, 40, 60, 200]
AGE_LABELS = ["0-12", "13-19", "20-40", "41-60", "60+"]

CATEGORICAL_FEATURES = ["gender", "appointment_dow", "appointment_month", "age_group"]
NUMERIC_FEATURES = [
    "waiting_days", "prior_noshow_rate", "sms_received", "chronic_condition",
    "scholarship", "diabetes", "alcoholism", "handicap",
]


def add_waiting_days(df: pd.DataFrame) -> pd.DataFrame:
    """waiting_days: Days between scheduling and appointment."""
    df = df.copy()
    if "_waiting_days" in df.columns:
        df["waiting_days"] = df["_waiting_days"]
    else:
        df["waiting_days"] = (
            (pd.to_datetime(df["appointment_date"]) - pd.to_datetime(df["scheduled_date"]))
            .dt.days.clip(lower=0)
        )
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """appointment_dow: day of week. appointment_month: month of the appointment."""
    df = df.copy()
    appt_date = pd.to_datetime(df["appointment_date"])
    df["appointment_dow"] = appt_date.dt.day_name()
    df["appointment_month"] = appt_date.dt.month_name()
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """age_group: binned age (0-12, 13-19, 20-40, 41-60, 60+)."""
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df


def add_prior_noshow_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    prior_noshow_rate: patient's historical no-show rate PRIOR to this
    appointment. Computed using only appointments that occurred strictly
    before the current one (sorted by scheduled_date) to avoid label leakage,
    consistent with the FRD business rule that prior_noshow_rate is
    recalculated after every completed appointment.
    """
    df = df.copy()
    if "no_show" not in df.columns:
        df["prior_noshow_rate"] = 0.0
        return df

    df = df.sort_values(["patient_id", "scheduled_date"])
    grp = df.groupby("patient_id")["no_show"]
    cum_count = grp.cumcount()
    cum_noshows = grp.transform(lambda s: s.shift().fillna(0).cumsum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(cum_count > 0, cum_noshows / cum_count, 0.0)
    df["prior_noshow_rate"] = rate
    return df


def engineer_features(df: pd.DataFrame, encode: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Returns a model-ready DataFrame containing
    the engineered features (and the `no_show` target, if present).

    Set encode=False to inspect raw (un-one-hot-encoded) feature values, e.g.
    in notebooks or tests.
    """
    df = add_waiting_days(df)
    df = add_date_features(df)
    df = add_age_group(df)
    df = add_prior_noshow_rate(df)

    # Keep optional Kaggle health/social features when present; API scoring
    # supplies defaults for them when a live request does not contain them.
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    keep_cols = ["appointment_id"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if "no_show" in df.columns:
        keep_cols.append("no_show")
    result = df[keep_cols].copy()

    if encode:
        result = pd.get_dummies(result, columns=CATEGORICAL_FEATURES, drop_first=True)

    logger.info("Engineered features: %d rows, %d columns", *result.shape)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Engineer features from cleaned appointments.")
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    args = parser.parse_args()

    cleaned = pd.read_csv(args.input_csv)
    features = engineer_features(cleaned)
    features.to_csv(args.output, index=False)
    print(f"Saved {len(features)} rows x {features.shape[1]} cols to {args.output}")
