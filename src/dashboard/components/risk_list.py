
"""
Risk-ranked appointment list component (FR-RD-01..05, UC-01, UC-02).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


RISK_COLORS = {
    "high": "#e74c3c",
    "medium": "#f39c12",
    "low": "#2ecc71",
}


def _load_contact_log(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame(
        columns=["appointment_id", "contacted_at", "outcome"]
    )


def _save_contact_log(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def render_risk_list(
    df: pd.DataFrame,
    contact_log_path: Path,
) -> None:

    st.subheader("Appointments Ranked by No-Show Risk")

    # Make appointment dates timezone-naive so they can be
    # compared safely with Streamlit date_input values.
    df = df.copy()

    df["appointment_date"] = pd.to_datetime(
        df["appointment_date"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    col1, col2, col3 = st.columns(3)

    with col1:
        date_range = st.date_input(
            "Appointment date range",
            value=(
                df["appointment_date"].min().date(),
                df["appointment_date"].max().date(),
            ),
        )

    with col2:
        tiers = st.multiselect(
            "Risk tier",
            ["high", "medium", "low"],
            default=["high", "medium", "low"],
        )

    with col3:
        doctor_options = (
            sorted(df["doctor"].dropna().unique())
            if "doctor" in df.columns
            else []
        )

        doctor = (
            st.multiselect(
                "Doctor",
                doctor_options,
                default=doctor_options,
            )
            if doctor_options
            else None
        )

    # Filter by risk tier
    filtered = df[df["risk_tier"].isin(tiers)].copy()

    # Filter by appointment date
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        filtered = filtered[
            (filtered["appointment_date"] >= start)
            & (filtered["appointment_date"] <= end)
        ].copy()

    # Filter by doctor
    if doctor:
        filtered = filtered[
            filtered["doctor"].isin(doctor)
        ].copy()

    # Highest-risk appointments first
    filtered = filtered.sort_values(
        "no_show_probability",
        ascending=False,
    )

    # Load contact history
    contact_log = _load_contact_log(contact_log_path)

    filtered = filtered.merge(
        contact_log,
        on="appointment_id",
        how="left",
    )

    def _style_row(row):
        color = RISK_COLORS.get(
            row["risk_tier"],
            "#ffffff",
        )

        return [
            f"background-color: {color}22"
        ] * len(row)

    display_cols = [
        c
        for c in [
            "appointment_id",
            "patient_id",
            "appointment_date",
            "doctor",
            "no_show_probability",
            "risk_tier",
            "outcome",
        ]
        if c in filtered.columns
    ]

    # Display only the highest-risk 100 appointments.
    # This prevents Pandas Styler from exceeding
    # Streamlit's maximum rendered-cell limit.
    display_df = filtered[display_cols].head(100).copy()

    st.caption(
        f"Showing top {len(display_df)} highest-risk appointments "
        f"out of {len(filtered):,} matching appointments."
    )

    st.dataframe(
        display_df.style.apply(
            _style_row,
            axis=1,
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("**Log a contact outcome (UC-02)**")

    c1, c2, c3 = st.columns(3)

    with c1:
        appt_id = st.selectbox(
            "Appointment",
            (
                filtered["appointment_id"].tolist()
                if not filtered.empty
                else []
            ),
        )

    with c2:
        outcome = st.selectbox(
            "Outcome",
            [
                "confirmed",
                "rescheduled",
                "no response",
            ],
        )

    with c3:
        if st.button("Save outcome") and appt_id:

            new_row = pd.DataFrame(
                [
                    {
                        "appointment_id": appt_id,
                        "contacted_at": pd.Timestamp.utcnow(),
                        "outcome": outcome,
                    }
                ]
            )

            contact_log = pd.concat(
                [
                    contact_log[
                        contact_log["appointment_id"] != appt_id
                    ],
                    new_row,
                ],
                ignore_index=True,
            )

            _save_contact_log(
                contact_log_path,
                contact_log,
            )

            st.success(
                f"Logged '{outcome}' for {appt_id}. "
                "Refresh to see the update."
            )
