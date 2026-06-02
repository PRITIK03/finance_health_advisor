"""
Alerts & Notifications Page Module
Smart alerts based on financial thresholds.
"""
import streamlit as st
import numpy as np
import pandas as pd
from components import UIComponents


def render_alerts(users_df, monthly_df):
    """Render the Smart Alerts & Notifications page."""
    UIComponents.page_header(
        "Smart Alerts & Notifications",
        "Proactive alerts when spending, savings, or risk thresholds are breached.",
        icon="🔔"
    )
    UIComponents.info_box("Proactive alerts when spending, savings, or risk thresholds are breached.", icon="⚠️")

    st.subheader("Critical Alerts (Demo)")
    alert_users = users_df[
        (users_df['monthly_savings'] / users_df['monthly_income'].replace(0, np.nan) < 0.05) |
        (users_df['monthly_expenses'] / users_df['monthly_income'].replace(0, np.nan) > 0.85)
    ]

    if not alert_users.empty:
        for _, row in alert_users.head(5).iterrows():
            st.error(f"User {row['user_id']}: Very low savings rate or dangerously high expense ratio!")
    else:
        st.success("No critical alerts in current page.")
