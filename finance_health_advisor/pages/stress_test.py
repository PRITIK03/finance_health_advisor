"""
Stress Test Page Module
Financial Stress Simulator for testing resilience to various shocks.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components import UIComponents
from preprocessing import calculate_financial_stress_test


def render_stress_test(users_df, monthly_df, visualizer):
    """Render the Financial Stress Simulator page."""
    UIComponents.page_header(
        "Financial Stress Simulator",
        "Simulate various financial shocks to understand their impact on your financial resilience and net worth over time.",
        icon="🚨"
    )

    UIComponents.info_box("Simulate various financial shocks to understand their impact on your financial resilience and net worth over time.")

    user_row = UIComponents.user_selector(users_df, key="stress_user")

    with st.container(border=True):
        st.markdown("<p class='card-title'>⚙️ Stress Parameters</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            scenario_name = st.selectbox(
                "Select Stress Scenario:",
                ["Job Loss (6 months)", "Medical Emergency", "Market Downturn (20% loss)", "Unexpected Large Expense"]
            )
        with col2:
            simulation_months = st.slider(
                "Simulation Duration (Months)",
                6, 60, 24
            )

    stress_results = calculate_financial_stress_test(
        user_row,
        scenario_name=scenario_name,
        simulation_months=simulation_months
    )

    UIComponents.metric_row([
        {"label": "Initial Liquid Assets", "value": f"${stress_results['initial_liquid_assets']:,.0f}"},
        {"label": "Final Liquid Assets", "value": f"${stress_results['final_liquid_assets']:,.0f}"},
        {"label": "Initial Net Worth", "value": f"${stress_results['initial_net_worth']:,.0f}"},
        {"label": "Final Net Worth", "value": f"${stress_results['final_net_worth']:,.0f}"},
    ], columns=4)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 2.5])

    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>🛟 Resilience Summary</p>", unsafe_allow_html=True)
            st.write(f"**Scenario:** {stress_results['scenario_name']}")
            st.info(stress_results['message'])
            st.write(f"**Outcome:** {stress_results['outcome']}")
            st.write(f"**Severity:** {stress_results['severity']}")
            st.write(f"**Months of Resilience:** {stress_results['months_of_resilience']} out of {simulation_months}")

            resilience_score = (stress_results['months_of_resilience'] / simulation_months) * 100
            st.plotly_chart(
                visualizer.create_gauge_chart(
                    min(100, resilience_score),
                    "Financial Resilience Score",
                    target=100
                ),
                use_container_width=True
            )

    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Financial Trajectory Over Time</p>", unsafe_allow_html=True)

            fig = go.Figure()
            months = list(range(simulation_months + 1))

            fig.add_trace(go.Scatter(
                x=months,
                y=stress_results['liquid_assets_history'],
                mode='lines',
                name='Liquid Assets',
                line=dict(color='#3b82f6')
            ))
            fig.add_trace(go.Scatter(
                x=months,
                y=stress_results['investments_history'],
                mode='lines',
                name='Investments',
                line=dict(color='#10b981')
            ))
            fig.add_trace(go.Scatter(
                x=months,
                y=stress_results['net_worth_history'],
                mode='lines',
                name='Net Worth',
                line=dict(color='#f59e0b', dash='dash')
            ))

            UIComponents.plotly_defaults(fig, height=350)
            fig.update_layout(
                xaxis_title="Months",
                yaxis_title="Amount ($)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("<p class='card-title'>💡 Key Takeaways</p>", unsafe_allow_html=True)
        severity = stress_results['severity']
        outcome = stress_results['outcome']
        if severity == "High":
            st.error(f"**Critical:** {outcome} Immediate action is needed to build reserves or reduce liabilities.")
        elif severity == "Medium":
            st.warning(f"**Warning:** {outcome} Consider strengthening your financial buffer to withstand longer or more severe shocks.")
        else:
            st.success(f"**Good:** {outcome} Your financial position appears resilient to this scenario.")

        st.markdown("---")
        st.write("This simulation provides a simplified view. For personalized advice, consult a financial expert.")
