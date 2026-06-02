"""
Goal Planner Page Module
Financial Goal Roadmap with timeline projection and optimization tips.
"""
import streamlit as st
import pandas as pd
import numpy as np
from components import UIComponents


def render_goal_planner(users_df, monthly_df):
    """Render the Financial Goal Roadmap page."""
    UIComponents.page_header(
        "Financial Goal Roadmap",
        "Plan your financial milestones and see when you'll reach them based on your current savings profile.",
        icon="🎯"
    )
    UIComponents.info_box("Plan your financial milestones and see when you'll reach them based on your current savings profile.")

    user_row = UIComponents.user_selector(users_df, key="goal_user")
    monthly_data = monthly_df[monthly_df['user_id'] == user_row['user_id']]
    avg_savings = monthly_data['savings'].mean() if not monthly_data.empty else 0

    with st.container(border=True):
        st.markdown("<p class='card-title'>⚙️ Goal Parameters</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            target_amount = st.number_input("Target Amount ($)", min_value=1000, value=int(user_row['monthly_income'] * 6), step=1000)
        with col2:
            monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=int(avg_savings), step=100)
        with col3:
            expected_return = st.slider("Expected Annual Return (%)", 0.0, 15.0, 7.0, 0.5) / 100

    monthly_income_val = float(user_row['monthly_income']) if user_row['monthly_income'] > 0 else 1
    current_savings = float(user_row['monthly_savings'] * 12) if user_row['monthly_savings'] > 0 else float(monthly_income_val * 0.1)
    months_to_goal = 0
    temp_amount = current_savings

    if monthly_contribution > 0 or current_savings > 0:
        monthly_rate = (1 + expected_return) ** (1/12) - 1
        while temp_amount < target_amount and months_to_goal < 600:
            temp_amount = (temp_amount + monthly_contribution) * (1 + monthly_rate)
            months_to_goal += 1

    years_to_goal = months_to_goal / 12
    goal_date = pd.Timestamp.now() + pd.DateOffset(months=months_to_goal)
    progress_pct = min(100, (current_savings / target_amount) * 100) if target_amount > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 2, 1.5])

    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>Progress Overview</p>", unsafe_allow_html=True)
            st.metric("Current Savings", f"${current_savings:,.0f}")
            st.metric("Target Amount", f"${target_amount:,.0f}")
            st.metric("Progress", f"{progress_pct:.1f}%")
            st.progress(progress_pct / 100, text=f"{progress_pct:.1f}%")

    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>Timeline Projection</p>", unsafe_allow_html=True)
            if months_to_goal >= 600:
                st.error("Goal may be unreachable with current contribution rate")
            else:
                st.metric("Months to Goal", f"{months_to_goal}")
                st.metric("Years to Goal", f"{years_to_goal:.1f}")
                st.metric("Estimated Achievement", goal_date.strftime("%B %Y"))
                st.success(f"At ${monthly_contribution:,.0f}/month, you'll reach your goal by {goal_date.strftime('%B %Y')}!")

    with col3:
        with st.container(border=True):
            st.markdown("<p class='card-title'>Optimization Tips</p>", unsafe_allow_html=True)
            if monthly_contribution < target_amount * 0.02:
                st.warning("Consider increasing monthly contributions")
            if expected_return < 0.05:
                st.info("Higher returns could accelerate your timeline")
            if progress_pct < 10:
                st.info("Starting is the hardest part - keep going!")
            st.caption("Adjust parameters to see real-time updates")
