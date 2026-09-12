"""
Data validation utilities.

Implements the schema and data-quality checks defined in the FRD
(Section 8: Data Validation Rules) and referenced by the TRD's
Data Ingestion component (FR-DM-02).
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "appointment_id",
    "patient_id",
    "gender",
    "age",
    "scheduled_date",
    "appointment_date",
    "sms_received",
]

ACCEPTED_GENDERS = {"M", "F"}
MAX_AGE = 120


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed Kaggle/date-export formats robustly and normalize timezones."""
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed


def validate_schema(df: pd.DataFrame, training: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """
    Validate a raw appointments DataFrame against the FRD data validation rules.

    Parameters
    ----------
    df : pd.DataFrame
        Raw appointment records, already column-renamed to the canonical schema
        by `src.data.ingestion.standardize_columns`.
    training : bool
        If True, also require and validate the `no_show` target column.

    Returns
    -------
    (valid_df, errors) : tuple
        valid_df -> subset of rows that passed all checks
        errors   -> list of dicts describing every rejected/quarantined row:
                    {"row_index": int, "appointment_id": Any, "reason": str}
    """
    errors: list[dict] = []
    required = REQUIRED_COLUMNS + (["no_show"] if training else [])

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df["scheduled_date"] = _parse_dates(df["scheduled_date"])
    df["appointment_date"] = _parse_dates(df["appointment_date"])

    keep_mask = pd.Series(True, index=df.index)

    def reject(mask: pd.Series, reason: str) -> None:
        nonlocal keep_mask
        bad_rows = df[mask & keep_mask]
        for idx, row in bad_rows.iterrows():
            errors.append({
                "row_index": idx,
                "appointment_id": row.get("appointment_id"),
                "reason": reason,
            })
        keep_mask &= ~mask

    # Normalize to calendar dates before comparing order. The public Kaggle dataset
    # stores ScheduledDay with a timestamp while AppointmentDay is midnight.
    # Comparing raw timestamps would incorrectly reject valid same-day appointments.
    scheduled_day = df["scheduled_date"].dt.normalize()
    appointment_day = df["appointment_date"].dt.normalize()

    # Appointment ID must be unique per record
    dup_mask = df["appointment_id"].duplicated(keep="first")
    reject(dup_mask, "Duplicate appointment_id")

    # Normalize common numeric fields before validation.
    for col in ["age", "sms_received", "chronic_condition", "scholarship", "diabetes", "alcoholism", "handicap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Age: non-negative integer, reasonable upper bound (<= 120)
    # The public Kaggle dataset uses age=-1 for unknown age. Treat it as
    # missing so the preprocessing stage can impute it instead of rejecting
    # the otherwise valid appointment.
    df.loc[df["age"] == -1, "age"] = pd.NA
    bad_age = df["age"].notna() & ((df["age"] < 0) | (df["age"] > MAX_AGE))
    reject(bad_age, f"Age must be between 0 and {MAX_AGE}")

    # Dates must be valid; Appointment Date >= Scheduled Date
    bad_dates = df["scheduled_date"].isna() | df["appointment_date"].isna()
    reject(bad_dates, "Invalid scheduled_date or appointment_date")
    bad_order = (~bad_dates) & (appointment_day < scheduled_day)
    reject(bad_order, "appointment_date is before scheduled_date")

    # Gender must match an accepted set of values
    bad_gender = ~df["gender"].astype(str).str.upper().isin(ACCEPTED_GENDERS)
    reject(bad_gender, f"Gender must be one of {sorted(ACCEPTED_GENDERS)}")

    # SMS Received must be boolean only
    bad_sms = ~df["sms_received"].isin([0, 1, True, False])
    reject(bad_sms, "sms_received must be boolean (0/1/True/False)")

    if training:
        # The ingestion layer normalizes Kaggle Yes/No to 1/0. Keep this
        # validation tolerant of boolean/numeric representations too.
        target_numeric = pd.to_numeric(df["no_show"], errors="coerce")
        bad_target = target_numeric.isna() | ~target_numeric.isin([0, 1])
        reject(bad_target, "no_show target must be boolean (0/1/True/False) or Yes/No")
        df["no_show"] = target_numeric

    valid_df = df[keep_mask].reset_index(drop=True)
    return valid_df, errors
