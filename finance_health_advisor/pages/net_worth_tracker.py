"""
Net Worth Tracker page module
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_net_worth_tracker(users_df, monthly_df, visualizer):
    """Render the Net Worth Tracker page."""
    st.header("💰 Net Worth Tracker")
    st.info("Track your financial progress over time with detailed asset and liability breakdowns.")

    # User selector
    selected_user_id = st.selectbox(
        "Select User Profile",
        users_df['user_id'].unique(),
        format_func=lambda x: f"User {x}"
    )

    user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]
    user_monthly = monthly_df[monthly_df['user_id'] == selected_user_id].sort_values('month')

    if user_monthly.empty:
        st.warning("No monthly data available for this user.")
        return

    # Calculate net worth components using actual data columns
    # Assets: cumulative savings + investments (approximated from monthly data)
    # We build a progressive net worth based on savings accumulation
    user_monthly = user_monthly.copy()
    
    # Starting assets (approximated from user profile)
    # Use monthly_savings * 6 as initial emergency fund + some base
    initial_savings = float(user_row.get('monthly_savings', 0)) * 6
    initial_investments = float(user_row.get('monthly_investments', 0)) * 12 * 2  # 2 years of investments
    
    # Build cumulative assets over time
    user_monthly['cumulative_savings'] = initial_savings + user_monthly['savings'].cumsum()
    user_monthly['investments'] = initial_investments + user_monthly['monthly_investments'].cumsum() if 'monthly_investments' in user_monthly.columns else initial_investments
    user_monthly['assets'] = user_monthly['cumulative_savings'] + user_monthly['investments']
    
    # Liabilities: start with total_debt, reduce by loan payments
    initial_debt = float(user_row.get('total_debt', 0))
    monthly_loan_payment = float(user_row.get('monthly_loan_payments', 0))
    user_monthly['liabilities'] = initial_debt - (monthly_loan_payment * user_monthly['month'].cumsum())
    user_monthly['liabilities'] = user_monthly['liabilities'].clip(lower=0)
    
    user_monthly['net_worth'] = user_monthly['assets'] - user_monthly['liabilities']

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Net Worth", f"${user_monthly['net_worth'].iloc[-1]:,.0f}")
    with col2:
        st.metric("Total Assets", f"${user_monthly['assets'].iloc[-1]:,.0f}")
    with col3:
        st.metric("Total Liabilities", f"${user_monthly['liabilities'].iloc[-1]:,.0f}")
    with col4:
        st.metric("Net Worth Change", f"${user_monthly['net_worth'].iloc[-1] - user_monthly['net_worth'].iloc[0]:+,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Net Worth Trend", "📊 Asset Breakdown", "💳 Liability Analysis"])

    with tab1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Net Worth Over Time</p>", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=user_monthly['month'],
                y=user_monthly['net_worth'],
                mode='lines+markers',
                name='Net Worth',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                x=user_monthly['month'],
                y=user_monthly['assets'],
                mode='lines',
                name='Assets',
                line=dict(color='#10b981', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=user_monthly['month'],
                y=-user_monthly['liabilities'],
                mode='lines',
                name='Liabilities',
                line=dict(color='#ef4444', width=2, dash='dash')
            ))

            fig.update_layout(
                template="plotly_white",
                xaxis_title="Month",
                yaxis_title="Amount ($)",
                hovermode="x unified",
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Asset Composition</p>", unsafe_allow_html=True)

            latest_assets = pd.Series({
                'Liquid Savings': user_monthly['cumulative_savings'].iloc[-1],
                'Investments': user_monthly['investments'].iloc[-1]
            })
            if latest_assets.sum() > 0:
                fig = px.pie(
                    values=latest_assets.values,
                    names=latest_assets.index,
                    hole=0.6,
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No asset data available for this period.")

    with tab3:
        with st.container(border=True):
            st.markdown("<p class='card-title'>💳 Liability Trend</p>", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=user_monthly['month'],
                y=user_monthly['liabilities'],
                name='Liabilities',
                marker_color='#ef4444'
            ))
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Month",
                yaxis_title="Liability Amount ($)",
                margin=dict(t=30, b=30, l=30, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<p class='card-title'>📋 Monthly Snapshot</p>", unsafe_allow_html=True)
        st.dataframe(
            user_monthly[['month', 'assets', 'liabilities', 'net_worth']].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )