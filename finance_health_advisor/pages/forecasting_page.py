"""
Forecasting Page Module
Savings Forecasting with Gradient Boosting regressor results.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components import UIComponents
from visualizations import FinancialVisualizer


def render_forecasting(users_df, monthly_df, visualizer, results):
    """Render the Savings Forecasting page."""
    UIComponents.page_header(
        "Savings Forecasting",
        "Gradient Boosting regressor predicts future savings potential based on spending habits and income stability.",
        icon="📈"
    )
    UIComponents.info_box("Gradient Boosting regressor predicts future savings potential based on spending habits and income stability.")

    UIComponents.metric_row([
        {"label": "Model Variance (R²)", "value": f"{results['forecasting']['test_metrics']['r2']:.2f}"},
        {"label": "Avg Prediction Error", "value": f"${results['forecasting']['test_metrics']['mae']:.0f}"},
        {"label": "RMSE", "value": f"${results['forecasting']['test_metrics']['rmse']:.0f}"},
    ], columns=3)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Trends", "📊 Segment Analysis"])

    with tab1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Monthly Savings Trends & Projection</p>", unsafe_allow_html=True)
            
            monthly_avg = monthly_df.groupby('month').agg({
                'income': 'mean',
                'expenses': 'mean',
                'savings': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_avg['month'],
                y=monthly_avg['savings'],
                mode='lines+markers',
                name='Actual Savings',
                line=dict(color='#3b82f6', width=4, shape='spline'),
                marker=dict(size=10, borderwidth=2, color='white', line=dict(color='#3b82f6'))
            ))

            z = np.polyfit(monthly_avg['month'], monthly_avg['savings'], 2)
            p = np.poly1d(z)
            x_trend = np.arange(1, 13)
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='AI Forecast Trend',
                line=dict(color='#f43f5e', dash='dash', width=2)
            ))

            UIComponents.plotly_defaults(fig, height=380)
            fig.update_layout(
                xaxis_title='Month of Year',
                yaxis_title='Average Savings ($)',
                xaxis=dict(tickmode='linear', tick0=1, dtick=1, gridcolor='#f1f5f9'),
                yaxis=dict(gridcolor='#f1f5f9'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Savings Performance by Risk Profile</p>", unsafe_allow_html=True)
            
            savings_by_risk = monthly_df.groupby('risk_label')['savings'].mean().sort_values()
            
            import plotly.express as px
            fig = px.bar(
                x=savings_by_risk.index,
                y=savings_by_risk.values,
                labels={'x': 'Risk Category', 'y': 'Average Savings ($)'},
                color=savings_by_risk.values,
                color_continuous_scale='RdYlGn'
            )
            UIComponents.plotly_defaults(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)
