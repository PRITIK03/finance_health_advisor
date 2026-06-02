"""
Data Visualization Dashboard Page Module
Interactive visualizations for deep financial data exploration.
"""
import streamlit as st
import plotly.graph_objects as go
from components import UIComponents


def render_data_visualization(users_df, monthly_df, visualizer):
    """Render the Data Visualization Dashboard page."""
    UIComponents.page_header(
        "Comprehensive Data Visualization Dashboard",
        "Interactive visualizations for deep financial data exploration.",
        icon="📈"
    )
    UIComponents.info_box("Interactive visualizations for deep financial data exploration.", icon="📊")

    tabs = st.tabs(["Distributions", "Correlations", "Trends", "Employment", "Anomalies"])

    with tabs[0]:
        st.plotly_chart(visualizer.create_distribution_plots()['distribution_plots'], use_container_width=True)

    with tabs[1]:
        st.plotly_chart(visualizer.create_correlation_heatmap(), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(visualizer.create_time_series_plot(), use_container_width=True)

    with tabs[3]:
        st.plotly_chart(visualizer.create_employment_analysis(), use_container_width=True)

    with tabs[4]:
        st.plotly_chart(visualizer.create_anomaly_analysis_plot(), use_container_width=True)
