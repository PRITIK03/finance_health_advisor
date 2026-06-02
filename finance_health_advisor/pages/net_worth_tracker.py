"""
Net Worth Tracker page module
"""
import streamlit as st
import pandas as pd
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

    # Calculate net worth components
    user_monthly['assets'] = (
        user_monthly['Savings'] +
        user_monthly['Investments'] +
        user_monthly['Retirement']
    )
    user_monthly['liabilities'] = user_monthly['total_debt']
    user_monthly['net_worth'] = user_monthly['assets'] - user_monthly['liabilities']

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Current Net Worth", f"${user_monthly['net_worth'].iloc[-1]:,.0f}")
    with col2: st.metric("Total Assets", f"${user_monthly['assets'].iloc[-1]:,.0f}")
    with col3: st.metric("Total Liabilities", f"${user_monthly['liabilities'].iloc[-1]:,.0f}")
    with col4: st.metric("Net Worth Change", f"${user_monthly['net_worth'].iloc[-1] - user_monthly['net_worth'].iloc[0]:+,.0f}")

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

            latest_assets = user_monthly[['Savings', 'Investments', 'Retirement']].iloc[-1]
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