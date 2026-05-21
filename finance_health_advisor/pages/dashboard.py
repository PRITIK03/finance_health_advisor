"""
Dashboard page module
"""
import streamlit as st
import pandas as pd
import numpy as np

from visualizations import generate_summary_statistics
from recommendations import RecommendationsEngine


def render_dashboard(users_df, monthly_df, recommendations_engine, visualizer):
    """Render the main Dashboard Overview page."""
    st.header("Financial Executive Summary")
    stats = generate_summary_statistics(users_df, monthly_df)

    # Top Summary Cards
    top_col1, top_col2, top_col3, top_col4 = st.columns(4)
    with top_col1:
        st.metric("Avg Health Score", f"{stats['avg_health_score']:.1f}")
    with top_col2:
        st.metric("Avg Income", f"${stats['avg_income']:,.0f}")
    with top_col3:
        st.metric("Avg Savings", f"${stats['avg_savings']:,.0f}")
    with top_col4:
        st.metric("Avg Credit Score", f"{stats['avg_credit_score']:.0f}")

    st.write("")

    # Personalized Insights
    st.subheader("🤖 Personalized Insights")
    sample_user = users_df.iloc[0].to_dict() if not users_df.empty else {}
    budget_recs = recommendations_engine.get_budget_recommendations(sample_user) if sample_user else []
    debt_recs = recommendations_engine.get_debt_recommendations(sample_user) if sample_user else []
    for rec in (budget_recs[:1] + debt_recs[:1]):
        color = "#fde68a" if rec['status'] in ["warning", "moderate"] else ("#fca5a5" if rec['status'] == "critical" else ("#bbf7d0" if rec['status'] in ["good", "excellent"] else "#bae6fd"))
        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 8px; margin-bottom: 8px;'><b>{rec['category']}:</b> {rec['message']}<br><i>{rec['suggestion']}</i></div>", unsafe_allow_html=True)

    # Recent Alerts
    st.subheader("🔔 Recent Alerts & Notifications")
    alert_users = users_df[(users_df['monthly_savings'] / users_df['monthly_income'].replace(0, np.nan) < 0.1) | (users_df['monthly_expenses'] / users_df['monthly_income'].replace(0, np.nan) > 0.8)]
    if not alert_users.empty:
        for _, row in alert_users.head(3).iterrows():
            st.warning(f"User {row['user_id']}: Low savings rate or high expenses detected.")
    else:
        st.success("No critical alerts at this time.")

    # Metric cards with Gauge
    col1, col2 = st.columns([1.5, 2.5])
    with col1:
        with st.container(border=True):
            st.plotly_chart(visualizer.create_gauge_chart(stats['avg_health_score'], "System Health Index"), use_container_width=True)
            st.markdown(f"<div style='text-align: center; color: #64748b;'>Average score across <b>{stats['total_users']:,}</b> users</div>", unsafe_allow_html=True)
    with col2:
        sub_col1, sub_col2 = st.columns(2)
        monthly_income_trend = monthly_df.groupby('month')['income'].mean().tolist()
        monthly_savings_trend = monthly_df.groupby('month')['savings'].mean().tolist()
        with sub_col1:
            with st.container(border=False):
                st.metric("Avg Monthly Income", f"${stats['avg_income']:,.0f}")
                st.plotly_chart(visualizer.create_sparkline(monthly_income_trend, color="#10b981"), use_container_width=True)
            with st.container(border=False):
                st.metric("Avg Monthly Savings", f"${stats['avg_savings']:,.0f}")
                st.plotly_chart(visualizer.create_sparkline(monthly_savings_trend, color="#3b82f6"), use_container_width=True)
        with sub_col2:
            st.metric("Avg Credit Score", f"{stats['avg_credit_score']:.0f}")
            st.metric("Data Sample Size", f"{stats['total_monthly_records']:,} mos")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Key Performance Indicators", "🔍 Behavioral Patterns", "💸 Subscription Audit", "📋 Dataset Snapshot"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📊 Risk & Health Distribution</p>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_risk_analysis_plot(), use_container_width=True)
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🍕 Spending Categories</p>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_spending_breakdown_chart(), use_container_width=True)
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Monthly Financial Trends</p>", unsafe_allow_html=True)
            st.plotly_chart(visualizer.create_time_series_plot(), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🔍 Income vs Expenses</p>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_income_expense_scatter(), use_container_width=True)
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>💼 Employment Analysis</p>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_employment_analysis(), use_container_width=True)
        with st.container(border=True):
            st.markdown("<p class='card-title'>🌡️ Feature Correlation</p>", unsafe_allow_html=True)
            st.plotly_chart(visualizer.create_correlation_heatmap(), use_container_width=True)

    with tab3:
        from preprocessing import calculate_subscription_audit
        st.info("💡 **Subscription Leakage Detection:** AI analyzes your recurring costs to identify low-usage services.")
        selected_user_id = st.selectbox("Select User for Audit", users_df['user_id'].unique(), key="sub_audit_user")
        audit_df, audit_summary = calculate_subscription_audit(monthly_df, selected_user_id)

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1: st.metric("Active Services", audit_summary['services_count'])
        with metric_col2: st.metric("Monthly Cost", f"${audit_summary['estimated_monthly_total']:,.0f}")
        with metric_col3: st.metric("Potential Savings", f"${audit_summary['potential_monthly_savings']:,.0f}/mo")
        with metric_col4: st.metric("High-Risk Leaks", audit_summary['high_risk_count'])

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.container(border=True):
                st.plotly_chart(visualizer.create_subscription_audit_chart(audit_df), use_container_width=True)
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🚨 Priority Actions</p>", unsafe_allow_html=True)
                risky_items = audit_df[audit_df['usage'].isin(['Low', 'Medium'])]
                if risky_items.empty:
                    st.success("No major subscription leaks detected.")
                else:
                    for _, row in risky_items.iterrows():
                        if row['usage'] == 'Low':
                            st.error(f"{row['name']}: {row['recommendation']}")
                        else:
                            st.warning(f"{row['name']}: {row['recommendation']}")
                st.markdown("---")
                st.write(f"**Annual Savings Potential:** `${audit_summary['potential_annual_savings']:,.0f}`")

        with st.container(border=True):
            st.markdown("<p class='card-title'>📋 Subscription Review Table</p>", unsafe_allow_html=True)
            st.dataframe(audit_df[['name', 'cost', 'annual_cost', 'usage', 'recommendation', 'potential_monthly_savings']], use_container_width=True)

    with tab4:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📋 Recent Financial Snapshot</p>", unsafe_allow_html=True)
            st.dataframe(users_df[['user_id', 'age', 'employment_type', 'monthly_income', 'monthly_expenses', 'financial_health_score']].head(20), use_container_width=True)
