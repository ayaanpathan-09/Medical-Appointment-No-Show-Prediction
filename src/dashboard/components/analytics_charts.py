"""
Reporting & Analytics component (FR-RA-01..05, UC-04).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models import registry


def render_analytics(df: pd.DataFrame) -> None:
    st.subheader("No-Show Rate Trends")

    if "no_show" in df.columns:
        trend = (
            df.set_index("appointment_date")["no_show"]
            .resample("W").mean().reset_index(name="no_show_rate")
        )
        fig = px.line(trend, x="appointment_date", y="no_show_rate", title="Weekly No-Show Rate")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Historical `no_show` outcomes are not present in this dataset; showing predicted risk instead.")
        trend = (
            df.set_index("appointment_date")["no_show_probability"]
            .resample("W").mean().reset_index(name="avg_predicted_risk")
        )
        fig = px.line(trend, x="appointment_date", y="avg_predicted_risk", title="Weekly Average Predicted Risk")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Breakdown by Segment (FR-RA-02)")
    breakdown_col = st.selectbox(
        "Break down by",
        [c for c in ["age_group", "gender", "appointment_dow", "sms_received"] if c in df.columns],
    )
    if breakdown_col:
        metric = "no_show" if "no_show" in df.columns else "no_show_probability"
        seg = df.groupby(breakdown_col)[metric].mean().reset_index()
        fig2 = px.bar(seg, x=breakdown_col, y=metric, title=f"{metric} by {breakdown_col}")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Model Performance (FR-RA-03)")
    prod_metrics = registry.get_production_metrics()
    if prod_metrics:
        cols = st.columns(5)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1_score", "roc_auc"]):
            col.metric(key.replace("_", " ").title(), f"{prod_metrics[key]:.3f}")
    else:
        st.info("No production model registered yet.")

    st.subheader("Intervention Effectiveness (FR-RA-05)")
    contact_log_path = Path("data/processed/contact_log.csv")
    if contact_log_path.exists() and "no_show" in df.columns:
        contacted = pd.read_csv(contact_log_path)["appointment_id"].unique()
        df = df.copy()
        df["was_contacted"] = df["appointment_id"].isin(contacted)
        effect = df[df["risk_tier"] == "high"].groupby("was_contacted")["no_show"].mean().reset_index()
        fig3 = px.bar(effect, x="was_contacted", y="no_show",
                      title="No-Show Rate: Contacted vs. Non-Contacted High-Risk Patients")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No contact outcomes logged yet. Log outcomes in the Risk-Ranked List tab to populate this chart.")
