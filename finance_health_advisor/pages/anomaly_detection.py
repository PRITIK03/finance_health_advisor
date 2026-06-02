"""
Anomaly Detection Page Module
Isolation Forest anomaly detection for spending patterns.
"""
import streamlit as st
import pandas as pd
import numpy as np
from components import UIComponents


def render_anomaly_detection(users_df, monthly_df, visualizer):
    """Render the Anomaly Detection page."""
    UIComponents.page_header(
        "Anomaly Detection",
        "Isolation Forest algorithm identifies outliers in spending and savings that deviate from established user patterns.",
        icon="🚨"
    )
    UIComponents.info_box("Isolation Forest algorithm identifies outliers in spending and savings that deviate from established user patterns.")

    anomaly_count = monthly_df['ml_anomaly'].sum()
    anomaly_pct = anomaly_count / len(monthly_df) * 100

    UIComponents.metric_row([
        {"label": "Total Anomalies Detected", "value": f"{anomaly_count}"},
        {"label": "Anomaly Rate", "value": f"{anomaly_pct:.2f}%"},
    ], columns=2)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🚨 Visual Analysis", "📋 Detected Anomalies"])

    with tab1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>🚨 Spending & Savings Anomalies</p>", unsafe_allow_html=True)
            st.plotly_chart(visualizer.create_anomaly_analysis_plot(), use_container_width=True)

    with tab2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>🔍 Flagged Transactions (AI Insights)</p>", unsafe_allow_html=True)
            anomalies = monthly_df[monthly_df['ml_anomaly'] == 1]
            st.dataframe(
                anomalies[['user_id', 'month', 'income', 'expenses', 'savings', 'Entertainment', 'Shopping']].head(20),
                use_container_width=True
            )
