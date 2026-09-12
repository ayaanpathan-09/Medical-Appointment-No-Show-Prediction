"""
Risk Dashboard entry point (FRD Section 4.3 / TRD Section 2: Dashboard / Staff Workflow).

Run with:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.components.analytics_charts import render_analytics
from src.dashboard.components.risk_list import render_risk_list

st.set_page_config(page_title="No-Show Risk Dashboard", layout="wide")

PREDICTIONS_PATH = Path("data/processed/predictions.csv")
CONTACT_LOG_PATH = Path("data/processed/contact_log.csv")


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["appointment_date"])


def main() -> None:
    st.title("Medical Appointment No-Show Risk Dashboard")

    df = load_predictions(PREDICTIONS_PATH)
    if df.empty:
        st.warning(
            "No scored appointments found yet.\n\n"
            "1. Train a model: `python -m src.models.train data/processed/features.csv`\n"
            "2. Batch-score upcoming appointments via `POST /api/v1/predict/batch`\n"
            f"3. Save the results to `{PREDICTIONS_PATH}` (with columns: appointment_id, "
            "patient_id, appointment_date, no_show_probability, risk_tier, and optionally "
            "doctor / no_show)."
        )
        return

    tab_risk, tab_analytics = st.tabs(["Risk-Ranked List", "Analytics"])

    with tab_risk:
        render_risk_list(df, CONTACT_LOG_PATH)

    with tab_analytics:
        render_analytics(df)


if __name__ == "__main__":
    main()
