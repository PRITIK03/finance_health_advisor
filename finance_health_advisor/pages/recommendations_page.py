"""
Recommendations Page Module
AI-powered recommendations tailored to financial profile.
"""
import streamlit as st
import pandas as pd
from components import UIComponents


def render_recommendations(users_df, monthly_df, recommendations_engine, visualizer):
    """Render the Financial Recommendations page."""
    UIComponents.page_header(
        "Financial Recommendations",
        "AI-powered recommendations tailored to your financial profile based on behavioral ML analysis.",
        icon="💡"
    )
    UIComponents.info_box("AI-powered recommendations tailored to your financial profile based on behavioral ML analysis.")

    rec_type = st.radio(
        "Select Analysis Level:",
        ["👤 Individual User", "👥 Cohort Analysis"],
        horizontal=True
    )

    if rec_type == "👤 Individual User":
        user_id = st.selectbox(
            "Select a User Profile",
            users_df['user_id'].unique(),
            format_func=lambda x: f"User {x}"
        )
        user_row = users_df[users_df['user_id'] == user_id].iloc[0]

        col1, col2 = st.columns([1.5, 2.5])
        with col1:
            with st.container(border=True):
                st.plotly_chart(visualizer.create_gauge_chart(user_row['financial_health_score'], "Health Index"), use_container_width=True)
                st.markdown(f"<div style='text-align: center; font-weight: 700; color: #2563eb;'>{user_row['risk_label']} Risk</div>", unsafe_allow_html=True)

        with col2:
            sub_col1, sub_col2 = st.columns(2)
            user_monthly = monthly_df[monthly_df['user_id'] == user_id].sort_values('month')
            user_income_trend = user_monthly['income'].tolist()
            user_savings_trend = user_monthly['savings'].tolist()

            with sub_col1:
                with st.container(border=False):
                    st.metric("Monthly Income", f"${user_row['monthly_income']:,.0f}")
                    if user_income_trend:
                        st.plotly_chart(visualizer.create_sparkline(user_income_trend, color="#10b981", height=40), use_container_width=True)
                with st.container(border=False):
                    st.metric("Monthly Savings", f"${user_row['monthly_savings']:,.0f}")
                    if user_savings_trend:
                        st.plotly_chart(visualizer.create_sparkline(user_savings_trend, color="#3b82f6", height=40), use_container_width=True)

            with sub_col2:
                st.metric("Credit Score", f"{user_row['credit_score']:.0f}")
                st.metric("Expense Ratio", f"{user_row['expense_ratio']:.1%}")

        st.markdown("<br>", unsafe_allow_html=True)

        user_recs = recommendations_engine.generate_user_recommendations(user_id)
        diversification_profile = recommendations_engine.get_diversification_profile(user_row.to_dict())

        UIComponents.recommendations_tabs(user_recs, ["💰 Budget", "💳 Debt", "🏦 Savings", "📈 Investments"])

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Investment Strategy</p>", unsafe_allow_html=True)

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Risk Tolerance", diversification_profile['risk_tolerance'])
            with metric_col2:
                st.metric("Diversification Score", f"{diversification_profile['diversification_score']:.0f}/100")
            with metric_col3:
                st.metric("Largest Gap", diversification_profile['largest_gap_asset'])

            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(visualizer.create_diversification_radar(diversification_profile), use_container_width=True)
            with col2:
                st.markdown("**Investment Performance Insights:**")
                for rec in user_recs['investments']:
                    UIComponents.recommendation_item(rec)

            col3, col4 = st.columns([1.2, 0.8])
            with col3:
                st.plotly_chart(visualizer.create_diversification_gap_chart(diversification_profile), use_container_width=True)
            with col4:
                allocation_df = pd.DataFrame({
                    'Asset Class': list(diversification_profile['current_allocation'].keys()),
                    'Current %': list(diversification_profile['current_allocation'].values()),
                    'Target %': [diversification_profile['target_allocation'][k] for k in diversification_profile['current_allocation'].keys()],
                    'Gap %': [diversification_profile['gap_by_asset'][k] for k in diversification_profile['current_allocation'].keys()]
                })
                st.dataframe(allocation_df, use_container_width=True, hide_index=True)

            st.info(diversification_profile['rebalance_hint'])

    else:
        cluster_filter = st.selectbox(
            "Filter by Cluster Profile",
            ['All'] + list(range(5))
        )
        risk_filter_cohort = st.selectbox(
            "Filter by Risk Level",
            ['All', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
        )

        cluster_val = None if cluster_filter == 'All' else cluster_filter
        risk_val = None if risk_filter_cohort == 'All' else risk_filter_cohort

        cohort_recs = recommendations_engine.generate_cohort_recommendations(cluster_val, risk_val)

        if cohort_recs['cohort_size'] > 0:
            stats = cohort_recs['cohort_stats']
            UIComponents.metric_row([
                {"label": "Cohort Size", "value": f"{cohort_recs['cohort_size']:,}"},
                {"label": "Avg Income", "value": f"${stats['avg_income']:,.0f}"},
                {"label": "Avg Savings", "value": f"${stats['avg_savings']:,.0f}"},
                {"label": "Avg Health Score", "value": f"{stats['avg_health_score']:.1f}/100"},
            ], columns=4)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"✨ **Cohort Summary:** {cohort_recs['summary']}")

            UIComponents.recommendations_tabs(cohort_recs['recommendations'], ["💰 Budget", "💳 Debt", "🏦 Savings", "📈 Investments"])
        else:
            st.warning("No users found in the selected cohort.")
