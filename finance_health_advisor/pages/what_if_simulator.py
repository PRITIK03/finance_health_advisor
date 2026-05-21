"""
What-If Budget Simulator Page
Interactive adjustments to spending with real-time impact on Financial Health Score.
"""
import streamlit as st
import pandas as pd
import numpy as np

from data_generator import calculate_financial_health_score


def render_what_if_simulator(users_df, monthly_df):
    st.header("🛠️ What-If Budget Simulator")
    st.info("Adjust your spending categories and instantly see the impact on your financial health score, savings, and risk level.")

    # User selection
    selected_user_id = st.selectbox(
        "Select User Profile",
        users_df['user_id'].unique(),
        format_func=lambda x: f"User {x}"
    )
    user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0].copy()

    # Get average monthly spending breakdown for this user
    user_monthly = monthly_df[monthly_df['user_id'] == selected_user_id]
    spending_cols = ['Housing', 'Transportation', 'Food', 'Healthcare', 
                     'Entertainment', 'Shopping', 'Education', 'Subscriptions',
                     'Insurance', 'Miscellaneous']

    current_spending = user_monthly[spending_cols].mean().to_dict()
    current_expenses = float(user_row['monthly_expenses'])
    current_savings = float(user_row['monthly_savings'])
    current_health = float(user_row['financial_health_score'])

    st.markdown("### Current Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monthly Income", f"${user_row['monthly_income']:,.0f}")
    with col2:
        st.metric("Current Expenses", f"${current_expenses:,.0f}")
    with col3:
        st.metric("Current Savings", f"${current_savings:,.0f}")
    with col4:
        st.metric("Health Score", f"{current_health:.1f}")

    st.markdown("---")
    st.markdown("### Adjust Your Spending")

    adjustments = {}
    cols = st.columns(2)
    i = 0
    for cat in spending_cols:
        with cols[i % 2]:
            current_val = current_spending.get(cat, 0)
            pct_change = st.slider(
                f"{cat} ({current_val:,.0f})",
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                format="%d%%",
                key=f"adj_{cat}"
            )
            adjustments[cat] = pct_change / 100.0
        i += 1

    # Calculate new values
    new_expenses = 0.0
    for cat in spending_cols:
        new_expenses += current_spending.get(cat, 0) * (1 + adjustments[cat])

    new_savings = user_row['monthly_income'] - new_expenses
    new_savings = max(0, new_savings)

    # Build profile for health score calculation
    new_profile = {
        'monthly_income': user_row['monthly_income'],
        'monthly_expenses': new_expenses,
        'monthly_savings': new_savings,
        'monthly_investments': user_row['monthly_investments'],
        'credit_score': user_row['credit_score'],
        'total_debt': user_row['total_debt']
    }
    new_health = calculate_financial_health_score(new_profile)

    delta_health = new_health - current_health
    delta_savings = new_savings - current_savings
    delta_expenses = new_expenses - current_expenses

    st.markdown("---")
    st.markdown("### Projected Impact")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("New Monthly Expenses", f"${new_expenses:,.0f}", delta=f"{delta_expenses:+,.0f}")
    with m2:
        color = "normal" if delta_savings >= 0 else "inverse"
        st.metric("New Monthly Savings", f"${new_savings:,.0f}", delta=f"{delta_savings:+,.0f}")
    with m3:
        delta_color = "normal" if delta_health >= 0 else "inverse"
        st.metric("New Health Score", f"{new_health:.1f}", delta=f"{delta_health:+.1f}")

    # Visual comparison
    st.markdown("**Health Score Change**")
    if delta_health > 0:
        st.success(f"Excellent! Your financial health would improve by **{delta_health:.1f} points**.")
    elif delta_health < -5:
        st.error(f"Warning: This change would decrease your health score by **{abs(delta_health):.1f} points**.")
    else:
        st.info("Minor impact on your overall financial health.")

    # Quick recommendation
    if delta_savings > 200:
        st.balloons()
        st.success("This adjustment could significantly accelerate your wealth building and FIRE timeline!")
    elif new_savings < 0:
        st.error("This scenario would push you into negative savings. Not sustainable long-term.")

    st.caption("Note: Health score recalculation uses the same formula as the core model (savings rate, expense ratio, credit, investments, debt).")
