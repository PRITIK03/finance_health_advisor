"""
Financial Tips & Education Page Module
Evidence-based financial education modules.
"""
import streamlit as st
from components import UIComponents


def render_financial_tips():
    """Render the Financial Tips & Education page."""
    UIComponents.page_header(
        "Financial Literacy & Education Center",
        "Evidence-based financial education modules powered by behavioral insights.",
        icon="📚"
    )
    UIComponents.info_box("Evidence-based financial education modules powered by behavioral insights.", icon="🎓")

    with st.expander("💰 The 50/30/20 Budget Rule", expanded=True):
        st.markdown("""
        **Needs (50%)** — Housing, utilities, groceries, minimum debt payments, insurance.
        **Wants (30%)** — Dining out, entertainment, subscriptions, hobbies.
        **Savings & Debt (20%)** — Emergency fund, retirement, extra debt payments.
        """)

    with st.expander("🚨 Emergency Fund Essentials"):
        st.markdown("""
        Aim for **3-6 months** of essential expenses in a high-yield savings account.
        Start with a $1,000 starter fund if you're in debt, then build to full coverage.
        """)

    with st.expander("🔥 FIRE Movement Basics"):
        st.markdown("""
        **Financial Independence, Retire Early** — Save 50%+ of income to retire decades early.
        Rule of thumb: Multiply annual expenses by 25 (4% safe withdrawal rate).
        """)
