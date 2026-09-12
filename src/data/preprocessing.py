"""
Preprocessing module.

Cleans validated appointment data: deduplicates, imputes/flags missing
values, normalizes dates, and detects/caps outliers, per TRD Section 3.2.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate records on Appointment ID (FR-DM-03)."""
    before = len(df)
    df = df.drop_duplicates(subset=["appointment_id"], keep="first")
    logger.info("Deduplication removed %d rows", before - len(df))
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or flag missing values (e.g. missing age, missing SMS status)."""
    df = df.copy()
    if "age" in df.columns:
        df["age_missing"] = df["age"].isna().astype(int)
        df["age"] = df["age"].fillna(df["age"].median())
    if "sms_received" in df.columns:
        df["sms_received"] = df["sms_received"].fillna(0).astype(int)
    for col in ["chronic_condition", "scholarship", "diabetes", "alcoholism", "handicap"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    if "gender" in df.columns and df["gender"].isna().any():
        df["gender"] = df["gender"].fillna(df["gender"].mode().iloc[0])
    if "appointment_type" in df.columns:
        df["appointment_type"] = df["appointment_type"].fillna("Unknown")
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date fields into a consistent format."""
    df = df.copy()
    try:
        scheduled = pd.to_datetime(df["scheduled_date"], errors="coerce", format="mixed", utc=True)
        appointment = pd.to_datetime(df["appointment_date"], errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):
        scheduled = pd.to_datetime(df["scheduled_date"], errors="coerce", utc=True)
        appointment = pd.to_datetime(df["appointment_date"], errors="coerce", utc=True)
    df["scheduled_date"] = scheduled.dt.normalize()
    df["appointment_date"] = appointment.dt.normalize()
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and cap/flag outliers: negative waiting days, implausible ages.
    Waiting Days = Appointment Date - Scheduled Date (TRD Section 3.2).
    """
    df = df.copy()
    waiting_days = (df["appointment_date"] - df["scheduled_date"]).dt.days
    df["waiting_days_raw_flag"] = (waiting_days < 0).astype(int)
    df["_waiting_days"] = waiting_days.clip(lower=0)

    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=0, upper=120)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: dedup -> normalize dates -> impute -> cap outliers."""
    df = deduplicate(df)
    df = normalize_dates(df)
    df = impute_missing(df)
    df = cap_outliers(df)
    logger.info("Preprocessing complete: %d rows remain", len(df))
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean an ingested appointments CSV.")
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="data/processed/cleaned_appointments.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.input_csv)
    cleaned = clean_data(raw)
    cleaned.to_csv(args.output, index=False)
    print(f"Saved {len(cleaned)} cleaned rows to {args.output}")
