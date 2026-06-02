"""
Expense Categorization Page Module
AI-powered expense categorization and spending breakdown.
"""
import streamlit as st
import pandas as pd
from components import UIComponents


SPENDING_COLS = [
    'Housing', 'Transportation', 'Food', 'Healthcare',
    'Entertainment', 'Shopping', 'Education', 'Subscriptions',
    'Insurance', 'Miscellaneous'
]


def render_expense_classification(users_df, monthly_df, visualizer):
    """Render the Expense Categorization page."""
    UIComponents.page_header(
        "AI Expense Categorization",
        "Upload transactions or view auto-categorized spending from the synthetic dataset.",
        icon="💸"
    )
    UIComponents.info_box("Upload transactions or view auto-categorized spending from the synthetic dataset.", icon="🗂️")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Category Breakdown (Sample)")
        avg_spend = monthly_df[SPENDING_COLS].mean().sort_values(ascending=False)
        st.dataframe(avg_spend.reset_index().rename(columns={'index': 'Category', 0: 'Avg Monthly $'}))

    with col2:
        st.plotly_chart(visualizer.create_spending_breakdown_chart(), use_container_width=True)
