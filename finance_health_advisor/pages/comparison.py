"""
Comparison Mode Page Module
Side-by-side comparison of two users with metrics, charts, and winner analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
from components import UIComponents, plotly_defaults


def render_comparison(users_df, monthly_df, visualizer):
    """Render the Side-by-Side User Comparison page."""
    UIComponents.page_header(
        "Side-by-Side User Comparison",
        "Compare two financial profiles side-by-side to identify relative strengths and opportunities.",
        icon="👥"
    )
    UIComponents.info_box("Compare two financial profiles side-by-side to identify relative strengths and opportunities.")

    col1, col2 = st.columns(2)

    with col1:
        u1_id = st.selectbox("Select User A", users_df['user_id'].unique(), index=0)
        u1_data = users_df[users_df['user_id'] == u1_id].iloc[0]
        with st.container(border=True):
            st.plotly_chart(visualizer.create_gauge_chart(u1_data['financial_health_score'], f"User {u1_id} Health"), use_container_width=True)
            st.metric("Income", f"${u1_data['monthly_income']:,.0f}")
            st.metric("Savings", f"${u1_data['monthly_savings']:,.0f}")
            st.metric("Risk", u1_data['risk_label'])

    with col2:
        u2_id = st.selectbox("Select User B", users_df['user_id'].unique(), index=1)
        u2_data = users_df[users_df['user_id'] == u2_id].iloc[0]
        with st.container(border=True):
            st.plotly_chart(visualizer.create_gauge_chart(u2_data['financial_health_score'], f"User {u2_id} Health"), use_container_width=True)
            st.metric("Income", f"${u2_data['monthly_income']:,.0f}", delta=f"{u2_data['monthly_income'] - u1_data['monthly_income']:,.0f}")
            st.metric("Savings", f"${u2_data['monthly_savings']:,.0f}", delta=f"{u2_data['monthly_savings'] - u1_data['monthly_savings']:,.0f}")
            st.metric("Risk", u2_data['risk_label'])

    if u1_id == u2_id:
        st.warning("Select two different users to get a meaningful comparison.")
        return

    metrics_summary = [
        ("Monthly Income", u1_data['monthly_income'], u2_data['monthly_income'], True),
        ("Monthly Expenses", u1_data['monthly_expenses'], u2_data['monthly_expenses'], False),
        ("Monthly Savings", u1_data['monthly_savings'], u2_data['monthly_savings'], True),
        ("Monthly Investments", u1_data['monthly_investments'], u2_data['monthly_investments'], True),
        ("Credit Score", u1_data['credit_score'], u2_data['credit_score'], True),
        ("Health Score", u1_data['financial_health_score'], u2_data['financial_health_score'], True),
    ]

    comparison_rows = []
    user_a_wins = 0
    user_b_wins = 0
    for label, a_val, b_val, higher_is_better in metrics_summary:
        if a_val == b_val:
            leader = "Tie"
        elif (a_val > b_val and higher_is_better) or (a_val < b_val and not higher_is_better):
            leader = f"User {u1_id}"
            user_a_wins += 1
        else:
            leader = f"User {u2_id}"
            user_b_wins += 1

        gap_pct = ((b_val - a_val) / a_val * 100) if a_val not in [0, 0.0] else np.nan
        comparison_rows.append({
            "Metric": label,
            f"User {u1_id}": round(float(a_val), 2),
            f"User {u2_id}": round(float(b_val), 2),
            "Leader": leader,
            "Gap %": round(float(gap_pct), 1) if pd.notna(gap_pct) else None
        })

    winner = f"User {u1_id}" if user_a_wins > user_b_wins else f"User {u2_id}" if user_b_wins > user_a_wins else "Tie"

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<p class='card-title'>🏁 Head-to-Head Summary</p>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("User A Wins", user_a_wins)
        with col_b:
            st.metric("User B Wins", user_b_wins)
        with col_c:
            st.metric("Overall Leader", winner)

    col3, col4 = st.columns([1.4, 1.6])
    with col3:
        with st.container(border=True):
            st.markdown("<p class='card-title'>🎯 Comparative Radar Analysis</p>", unsafe_allow_html=True)
            st.plotly_chart(visualizer.create_peer_comparison_radar(u1_data, u2_data), use_container_width=True)
            st.caption("Comparing normalized metrics: Income, Expenses, Savings, Investments, Credit Score, Health Score.")
    with col4:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Metric Gap View</p>", unsafe_allow_html=True)
            st.plotly_chart(visualizer.create_user_comparison_chart(u1_data, u2_data), use_container_width=True)

    with st.container(border=True):
        st.markdown("<p class='card-title'>📋 Detailed Comparison Table</p>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
