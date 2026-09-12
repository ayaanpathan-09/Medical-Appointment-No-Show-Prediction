"""
Data ingestion module.

Loads historical / upcoming appointment data from CSV exports, standardizes
column names to the project's canonical schema, and logs every ingestion
run (FR-DM-01, FR-DM-04) as required by the TRD's Data Ingestion component.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src.data.validation import validate_schema
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Maps common raw column names -- including the public Kaggle
# "Medical Appointment No Shows" dataset -- to this project's canonical
# snake_case schema. Extend this map if your source system uses different
# column names.
COLUMN_MAP = {
    "PatientId": "patient_id",
    "PatientID": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "Age": "age",
    "ScheduledDay": "scheduled_date",
    "ScheduledDate": "scheduled_date",
    "AppointmentDay": "appointment_date",
    "AppointmentDate": "appointment_date",
    "SMS_received": "sms_received",
    "SMSReceived": "sms_received",
    "Hipertension": "chronic_condition",
    "Scholarship": "scholarship",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "Neighbourhood": "appointment_type",
    "Doctor": "doctor",
    "No-show": "no_show",
    "No_show": "no_show",
    "Noshow": "no_show",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to the canonical snake_case schema used across the project."""
    return df.rename(columns=COLUMN_MAP)


def _normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Yes/No and boolean no-show targets to integer 1/0."""
    if "no_show" not in df.columns:
        return df
    raw = df["no_show"]
    # Handle pandas object/string dtypes as well as mixed values consistently.
    text = raw.astype("string").str.strip().str.lower()
    mapped = text.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    numeric = pd.to_numeric(raw, errors="coerce")
    df["no_show"] = mapped.fillna(numeric)
    return df


def load_appointments_csv(filepath: str | Path, training: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Load a CSV of appointment records, standardize columns, and validate schema.

    Returns
    -------
    (valid_df, audit_log) : tuple
        valid_df  -> validated, canonical-schema DataFrame
        audit_log -> ingestion audit record (FR-DM-04): timestamp, row counts, errors
    """
    filepath = Path(filepath)
    raw_df = pd.read_csv(filepath)
    raw_count = len(raw_df)

    df = standardize_columns(raw_df)
    df = _normalize_target(df)

    valid_df, errors = validate_schema(df, training=training)

    audit_log = {
        "source_file": str(filepath),
        "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_record_count": raw_count,
        "valid_record_count": len(valid_df),
        "error_count": len(errors),
        "errors": errors,
    }

    logger.info(
        "Ingested %s: %d raw rows, %d valid, %d errors",
        filepath.name, raw_count, len(valid_df), len(errors),
    )
    return valid_df, audit_log


def save_processed(df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist a processed DataFrame to data/processed/ (or any given path)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d records to %s", len(df), output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a raw appointments CSV file.")
    parser.add_argument("input_csv", help="Path to raw CSV export")
    parser.add_argument("--output", default="data/processed/ingested_appointments.csv")
    parser.add_argument("--training", action="store_true", help="Require the no_show target column")
    args = parser.parse_args()

    valid, audit = load_appointments_csv(args.input_csv, training=args.training)
    save_processed(valid, args.output)
    print(audit)
