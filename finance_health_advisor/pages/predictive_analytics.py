"""
Predictive Analytics Page Module
Savings forecasts and Monte Carlo suite.
"""
import streamlit as st
from components import UIComponents


def render_predictive_analytics(users_df, monthly_df, visualizer, results):
    """Render the Predictive Analytics Suite page."""
    UIComponents.page_header(
        "Predictive Analytics Suite",
        "Advanced forecasting and scenario modeling (Monte Carlo + ML).",
        icon="🔮"
    )
    UIComponents.info_box("Advanced forecasting and scenario modeling (Monte Carlo + ML).", icon="🤖")

    st.subheader("Savings Forecast (Gradient Boosting)")
    st.metric("R² Score", f"{results['forecasting']['test_metrics']['r2']:.3f}")
    st.metric("RMSE", f"${results['forecasting']['test_metrics']['rmse']:.0f}")
    st.plotly_chart(visualizer.create_time_series_plot(), use_container_width=True)
