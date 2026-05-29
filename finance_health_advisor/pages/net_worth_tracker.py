"""
Net Worth Tracker page module
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _render_top_metrics(current_net_worth, current_assets, current_liabilities, net_worth_growth, net_worth_growth_pct):
    """Render the top metrics row."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Net Worth", f"${current_net_worth:,.0f}")
    with col2:
        st.metric("Total Assets", f"${current_assets:,.0f}")
    with col3:
        st.metric("Total Liabilities", f"${current_liabilities:,.0f}")
    with col4:
        st.metric("12-Month Growth", f"${net_worth_growth:,.0f}", f"{net_worth_growth_pct:+.1f}%")


def _render_trend_analysis_tab(net_worth_history, visualizer, net_worth_growth_pct):
    """Render the Trend Analysis tab content."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Net Worth Over Time</p>", unsafe_allow_html=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=net_worth_history['month'],
                y=net_worth_history['net_worth'],
                mode='lines+markers',
                name='Net Worth',
                line=dict(color='#10b981', width=3),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=net_worth_history['month'],
                y=net_worth_history['total_assets'],
                mode='lines',
                name='Total Assets',
                line=dict(color='#3b82f6', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=net_worth_history['month'],
                y=net_worth_history['total_liabilities'],
                mode='lines',
                name='Total Liabilities',
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
    
    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Growth Summary</p>", unsafe_allow_html=True)
            
            # Monthly growth rates
            monthly_growth = net_worth_history['net_worth'].pct_change() * 100
            avg_monthly_growth = monthly_growth.mean()
            
            st.metric("Avg Monthly Growth", f"{avg_monthly_growth:+.2f}%")
            st.metric("Best Month", f"${net_worth_history['net_worth'].max():,.0f}")
            st.metric("Lowest Month", f"${net_worth_history['net_worth'].min():,.0f}")
            
            # Growth gauge
            growth_score = min(100, max(0, net_worth_growth_pct + 50))  # Normalize to 0-100
            st.plotly_chart(
                visualizer.create_gauge_chart(
                    growth_score,
                    "Growth Momentum",
                    target=75
                ),
                use_container_width=True
            )
    
    with st.container(border=True):
        st.markdown("<p class='card-title'>📊 Monthly Change Breakdown</p>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=net_worth_history['month'],
            y=net_worth_history['assets_change'],
            name='Asset Change',
            marker_color='#10b981'
        ))
        
        fig.add_trace(go.Bar(
            x=net_worth_history['month'],
            y=-net_worth_history['liabilities_change'],
            name='Liability Change',
            marker_color='#ef4444'
        ))
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="Change ($)",
            hovermode="x unified",
            margin=dict(t=30, b=30, l=30, r=30),
            barmode='relative',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_asset_breakdown_tab(net_worth_history, current_assets, visualizer):
    """Render the Asset Breakdown tab content."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>🏦 Current Asset Allocation</p>", unsafe_allow_html=True)
            
            current_assets_breakdown = {
                'Cash & Savings': net_worth_history['cash_savings'].iloc[-1],
                'Investments': net_worth_history['investments'].iloc[-1],
                'Real Estate': net_worth_history['real_estate'].iloc[-1],
                'Retirement Accounts': net_worth_history['retirement'].iloc[-1],
                'Other Assets': net_worth_history['other_assets'].iloc[-1]
            }
            
            # Filter out zero values
            asset_breakdown = {k: v for k, v in current_assets_breakdown.items() if v > 0}
            
            fig = px.pie(
                values=list(asset_breakdown.values()),
                names=list(asset_breakdown.keys()),
                hole=0.6,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Asset Details</p>", unsafe_allow_html=True)
            
            for asset_type, value in current_assets_breakdown.items():
                percentage = (value / current_assets * 100) if current_assets > 0 else 0
                st.metric(asset_type, f"${value:,.0f}", f"{percentage:.1f}%")
    
    with st.container(border=True):
        st.markdown("<p class='card-title'>📈 Asset Growth Over Time</p>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
        asset_cols = ['cash_savings', 'investments', 'real_estate', 'retirement', 'other_assets']
        asset_names = ['Cash & Savings', 'Investments', 'Real Estate', 'Retirement', 'Other Assets']
        
        for col, name, color in zip(asset_cols, asset_names, colors):
            fig.add_trace(go.Scatter(
                x=net_worth_history['month'],
                y=net_worth_history[col],
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
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


def _render_liability_breakdown_tab(net_worth_history, current_liabilities, user_row, visualizer):
    """Render the Liability Breakdown tab content."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>💳 Current Liability Breakdown</p>", unsafe_allow_html=True)
            
            current_liabilities_breakdown = {
                'Mortgage': net_worth_history['mortgage'].iloc[-1],
                'Student Loans': net_worth_history['student_loans'].iloc[-1],
                'Credit Card Debt': net_worth_history['credit_card_debt'].iloc[-1],
                'Auto Loans': net_worth_history['auto_loans'].iloc[-1],
                'Other Debt': net_worth_history['other_debt'].iloc[-1]
            }
            
            # Filter out zero values
            liability_breakdown = {k: v for k, v in current_liabilities_breakdown.items() if v > 0}
            
            fig = px.pie(
                values=list(liability_breakdown.values()),
                names=list(liability_breakdown.keys()),
                hole=0.6,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Liability Details</p>", unsafe_allow_html=True)
            
            for liability_type, value in current_liabilities_breakdown.items():
                percentage = (value / current_liabilities * 100) if current_liabilities > 0 else 0
                st.metric(liability_type, f"${value:,.0f}", f"{percentage:.1f}%")
            
            st.markdown("---")
            
            # Debt-to-income ratio
            annual_income = user_row['monthly_income'] * 12
            dti_ratio = (current_liabilities / annual_income * 100) if annual_income > 0 else 0
            
            if dti_ratio < 30:
                dti_status = "Excellent"
                dti_color = "#10b981"
            elif dti_ratio < 50:
                dti_status = "Good"
                dti_color = "#f59e0b"
            else:
                dti_status = "High"
                dti_color = "#ef4444"
            
            st.metric("Debt-to-Income Ratio", f"{dti_ratio:.1f}%", dti_status)
    
    with st.container(border=True):
        st.markdown("<p class='card-title'>📉 Liability Reduction Over Time</p>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        liability_cols = ['mortgage', 'student_loans', 'credit_card_debt', 'auto_loans', 'other_debt']
        liability_names = ['Mortgage', 'Student Loans', 'Credit Card', 'Auto Loans', 'Other Debt']
        colors = ['#ef4444', '#f59e0b', '#8b5cf6', '#ec4899', '#6366f1']
        
        for col, name, color in zip(liability_cols, liability_names, colors):
            fig.add_trace(go.Scatter(
                x=net_worth_history['month'],
                y=net_worth_history[col],
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
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


def _render_goals_projections_tab(net_worth_history, current_net_worth, user_row, visualizer):
    """Render the Goals & Projections tab content."""
    st.info("🎯 Set your net worth goals and track your progress")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>🎯 Net Worth Goals</p>", unsafe_allow_html=True)
            
            # Goal inputs
            short_term_goal = st.number_input(
                "Short-term Goal (1 year) ($)",
                min_value=0,
                value=int(current_net_worth * 1.2),
                step=1000
            )
            
            medium_term_goal = st.number_input(
                "Medium-term Goal (5 years) ($)",
                min_value=0,
                value=int(current_net_worth * 2),
                step=5000
            )
            
            long_term_goal = st.number_input(
                "Long-term Goal (10 years) ($)",
                min_value=0,
                value=int(current_net_worth * 4),
                step=10000
            )
            
            # Calculate progress
            short_term_progress = (current_net_worth / short_term_goal * 100) if short_term_goal > 0 else 0
            medium_term_progress = (current_net_worth / medium_term_goal * 100) if medium_term_goal > 0 else 0
            long_term_progress = (current_net_worth / long_term_goal * 100) if long_term_goal > 0 else 0
            
            st.markdown("---")
            st.metric("Short-term Progress", f"{short_term_progress:.1f}%")
            st.progress(min(100, short_term_progress / 100))
            
            st.metric("Medium-term Progress", f"{medium_term_progress:.1f}%")
            st.progress(min(100, medium_term_progress / 100))
            
            st.metric("Long-term Progress", f"{long_term_progress:.1f}%")
            st.progress(min(100, long_term_progress / 100))
    
    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Projections</p>", unsafe_allow_html=True)
            
            # Projection parameters
            monthly_contribution = st.number_input(
                "Monthly Net Worth Contribution ($)",
                min_value=0,
                value=int(user_row['monthly_savings'] + user_row['monthly_investments']),
                step=100
            )
            
            annual_return = st.slider(
                "Expected Annual Return (%)",
                min_value=0.0,
                max_value=15.0,
                value=7.0,
                step=0.5
            ) / 100
            
            projection_years = st.slider(
                "Projection Years",
                min_value=1,
                max_value=30,
                value=10,
                step=1
            )
            
            # Calculate projection
            projected_net_worth = calculate_net_worth_projection(
                current_net_worth,
                monthly_contribution,
                annual_return,
                projection_years
            )
            
            st.markdown("---")
            st.metric("Projected Net Worth", f"${projected_net_worth:,.0f}")
            st.metric("Total Contributions", f"${monthly_contribution * 12 * projection_years:,.0f}")
            st.metric("Investment Returns", f"${projected_net_worth - current_net_worth - monthly_contribution * 12 * projection_years:,.0f}")
    
    with st.container(border=True):
        st.markdown("<p class='card-title'>📊 Projection Chart</p>", unsafe_allow_html=True)
        
        # Generate projection data
        projection_months = projection_years * 12
        months_range = list(range(projection_months + 1))
        projected_history = []
        
        temp_net_worth = current_net_worth
        monthly_return = (1 + annual_return) ** (1/12) - 1
        
        for month in months_range:
            projected_history.append(temp_net_worth)
            temp_net_worth = (temp_net_worth + monthly_contribution) * (1 + monthly_return)
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=net_worth_history['month'],
            y=net_worth_history['net_worth'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#10b981', width=3)
        ))
        
        # Projection
        fig.add_trace(go.Scatter(
            x=list(range(12, 12 + projection_months + 1)),
            y=projected_history,
            mode='lines',
            name='Projection',
            line=dict(color='#3b82f6', width=3, dash='dash')
        ))
        
        # Goal lines
        fig.add_hline(
            y=short_term_goal,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text="1-Year Goal",
            annotation_position="top right"
        )
        
        fig.add_hline(
            y=medium_term_goal,
            line_dash="dot",
            line_color="#8b5cf6",
            annotation_text="5-Year Goal",
            annotation_position="top right"
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="Net Worth ($)",
            hovermode="x unified",
            margin=dict(t=30, b=30, l=30, r=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🏦 Current Asset Allocation</p>", unsafe_allow_html=True)
                
                current_assets_breakdown = {
                    'Cash & Savings': net_worth_history['cash_savings'].iloc[-1],
                    'Investments': net_worth_history['investments'].iloc[-1],
                    'Real Estate': net_worth_history['real_estate'].iloc[-1],
                    'Retirement Accounts': net_worth_history['retirement'].iloc[-1],
                    'Other Assets': net_worth_history['other_assets'].iloc[-1]
                }
                
                # Filter out zero values
                asset_breakdown = {k: v for k, v in current_assets_breakdown.items() if v > 0}
                
                fig = px.pie(
                    values=list(asset_breakdown.values()),
                    names=list(asset_breakdown.keys()),
                    hole=0.6,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📊 Asset Details</p>", unsafe_allow_html=True)
                
                for asset_type, value in current_assets_breakdown.items():
                    percentage = (value / current_assets * 100) if current_assets > 0 else 0
                    st.metric(asset_type, f"${value:,.0f}", f"{percentage:.1f}%")
        
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Asset Growth Over Time</p>", unsafe_allow_html=True)
            
            fig = go.Figure()
            
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
            asset_cols = ['cash_savings', 'investments', 'real_estate', 'retirement', 'other_assets']
            asset_names = ['Cash & Savings', 'Investments', 'Real Estate', 'Retirement', 'Other Assets']
            
            for col, name, color in zip(asset_cols, asset_names, colors):
                fig.add_trace(go.Scatter(
                    x=net_worth_history['month'],
                    y=net_worth_history[col],
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2)
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
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>💳 Current Liability Breakdown</p>", unsafe_allow_html=True)
                
                current_liabilities_breakdown = {
                    'Mortgage': net_worth_history['mortgage'].iloc[-1],
                    'Student Loans': net_worth_history['student_loans'].iloc[-1],
                    'Credit Card Debt': net_worth_history['credit_card_debt'].iloc[-1],
                    'Auto Loans': net_worth_history['auto_loans'].iloc[-1],
                    'Other Debt': net_worth_history['other_debt'].iloc[-1]
                }
                
                # Filter out zero values
                liability_breakdown = {k: v for k, v in current_liabilities_breakdown.items() if v > 0}
                
                fig = px.pie(
                    values=list(liability_breakdown.values()),
                    names=list(liability_breakdown.keys()),
                    hole=0.6,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📊 Liability Details</p>", unsafe_allow_html=True)
                
                for liability_type, value in current_liabilities_breakdown.items():
                    percentage = (value / current_liabilities * 100) if current_liabilities > 0 else 0
                    st.metric(liability_type, f"${value:,.0f}", f"{percentage:.1f}%")
                
                st.markdown("---")
                
                # Debt-to-income ratio
                annual_income = user_row['monthly_income'] * 12
                dti_ratio = (current_liabilities / annual_income * 100) if annual_income > 0 else 0
                
                if dti_ratio < 30:
                    dti_status = "Excellent"
                    dti_color = "#10b981"
                elif dti_ratio < 50:
                    dti_status = "Good"
                    dti_color = "#f59e0b"
                else:
                    dti_status = "High"
                    dti_color = "#ef4444"
                
                st.metric("Debt-to-Income Ratio", f"{dti_ratio:.1f}%", dti_status)
        
        with st.container(border=True):
            st.markdown("<p class='card-title'>📉 Liability Reduction Over Time</p>", unsafe_allow_html=True)
            
            fig = go.Figure()
            
            liability_cols = ['mortgage', 'student_loans', 'credit_card_debt', 'auto_loans', 'other_debt']
            liability_names = ['Mortgage', 'Student Loans', 'Credit Card', 'Auto Loans', 'Other Debt']
            colors = ['#ef4444', '#f59e0b', '#8b5cf6', '#ec4899', '#6366f1']
            
            for col, name, color in zip(liability_cols, liability_names, colors):
                fig.add_trace(go.Scatter(
                    x=net_worth_history['month'],
                    y=net_worth_history[col],
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2)
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
    
    with tab4:
        st.info("🎯 Set your net worth goals and track your progress")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🎯 Net Worth Goals</p>", unsafe_allow_html=True)
                
                # Goal inputs
                short_term_goal = st.number_input(
                    "Short-term Goal (1 year) ($)",
                    min_value=0,
                    value=int(current_net_worth * 1.2),
                    step=1000
                )
                
                medium_term_goal = st.number_input(
                    "Medium-term Goal (5 years) ($)",
                    min_value=0,
                    value=int(current_net_worth * 2),
                    step=5000
                )
                
                long_term_goal = st.number_input(
                    "Long-term Goal (10 years) ($)",
                    min_value=0,
                    value=int(current_net_worth * 4),
                    step=10000
                )
                
                # Calculate progress
                short_term_progress = (current_net_worth / short_term_goal * 100) if short_term_goal > 0 else 0
                medium_term_progress = (current_net_worth / medium_term_goal * 100) if medium_term_goal > 0 else 0
                long_term_progress = (current_net_worth / long_term_goal * 100) if long_term_goal > 0 else 0
                
                st.markdown("---")
                st.metric("Short-term Progress", f"{short_term_progress:.1f}%")
                st.progress(min(100, short_term_progress / 100))
                
                st.metric("Medium-term Progress", f"{medium_term_progress:.1f}%")
                st.progress(min(100, medium_term_progress / 100))
                
                st.metric("Long-term Progress", f"{long_term_progress:.1f}%")
                st.progress(min(100, long_term_progress / 100))
        
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📈 Projections</p>", unsafe_allow_html=True)
                
                # Projection parameters
                monthly_contribution = st.number_input(
                    "Monthly Net Worth Contribution ($)",
                    min_value=0,
                    value=int(user_row['monthly_savings'] + user_row['monthly_investments']),
                    step=100
                )
                
                annual_return = st.slider(
                    "Expected Annual Return (%)",
                    min_value=0.0,
                    max_value=15.0,
                    value=7.0,
                    step=0.5
                ) / 100
                
                projection_years = st.slider(
                    "Projection Years",
                    min_value=1,
                    max_value=30,
                    value=10,
                    step=1
                )
                
                # Calculate projection
                projected_net_worth = calculate_net_worth_projection(
                    current_net_worth,
                    monthly_contribution,
                    annual_return,
                    projection_years
                )
                
                st.markdown("---")
                st.metric("Projected Net Worth", f"${projected_net_worth:,.0f}")
                st.metric("Total Contributions", f"${monthly_contribution * 12 * projection_years:,.0f}")
                st.metric("Investment Returns", f"${projected_net_worth - current_net_worth - monthly_contribution * 12 * projection_years:,.0f}")
        
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Projection Chart</p>", unsafe_allow_html=True)
            
            # Generate projection data
            projection_months = projection_years * 12
            months_range = list(range(projection_months + 1))
            projected_history = []
            
            temp_net_worth = current_net_worth
            monthly_return = (1 + annual_return) ** (1/12) - 1
            
            for month in months_range:
                projected_history.append(temp_net_worth)
                temp_net_worth = (temp_net_worth + monthly_contribution) * (1 + monthly_return)
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=net_worth_history['month'],
                y=net_worth_history['net_worth'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#10b981', width=3)
            ))
            
            # Projection
            fig.add_trace(go.Scatter(
                x=list(range(12, 12 + projection_months + 1)),
                y=projected_history,
                mode='lines',
                name='Projection',
                line=dict(color='#3b82f6', width=3, dash='dash')
            ))
            
            # Goal lines
            fig.add_hline(
                y=short_term_goal,
                line_dash="dot",
                line_color="#f59e0b",
                annotation_text="1-Year Goal",
                annotation_position="top right"
            )
            
            fig.add_hline(
                y=medium_term_goal,
                line_dash="dot",
                line_color="#8b5cf6",
                annotation_text="5-Year Goal",
                annotation_position="top right"
            )
            
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Month",
                yaxis_title="Net Worth ($)",
                hovermode="x unified",
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)


def generate_net_worth_history(user_row: pd.Series, months: int = 12) -> pd.DataFrame:
    """Generate synthetic net worth history for a user."""
    
    # Base values from user data
    monthly_income = float(user_row['monthly_income'])
    monthly_expenses = float(user_row['monthly_expenses'])
    monthly_savings = float(user_row['monthly_savings'])
    monthly_investments = float(user_row['monthly_investments'])
    total_debt = float(user_row['total_debt'])
    
    # Calculate asset components
    cash_savings_base = monthly_savings * 6  # 6 months of savings
    investments_base = monthly_investments * 24  # 2 years of investments
    real_estate_base = monthly_income * 12 * 2  # Rough estimate
    retirement_base = monthly_investments * 12
    other_assets_base = monthly_income * 3
    
    # Calculate liability components
    mortgage_base = total_debt * 0.6 if total_debt > 0 else 0
    student_loans_base = total_debt * 0.2 if total_debt > 0 else 0
    credit_card_debt_base = total_debt * 0.1 if total_debt > 0 else 0
    auto_loans_base = total_debt * 0.1 if total_debt > 0 else 0
    other_debt_base = total_debt * 0.0 if total_debt > 0 else 0
    
    # Generate historical data with some randomness
    np.random.seed(42)
    
    history = []
    for month in range(1, months + 1):
        # Add some month-to-month variation
        variation = np.random.normal(0, 0.02)  # 2% standard deviation
        
        cash_savings = cash_savings_base * (1 + variation * 0.5 + month * 0.01)
        investments = investments_base * (1 + variation * 1.5 + month * 0.02)
        real_estate = real_estate_base * (1 + variation * 0.3 + month * 0.005)
        retirement = retirement_base * (1 + variation * 1.0 + month * 0.015)
        other_assets = other_assets_base * (1 + variation * 0.2 + month * 0.008)
        
        # Liabilities decrease over time (debt paydown)
        mortgage = max(0, mortgage_base * (1 - month * 0.01))
        student_loans = max(0, student_loans_base * (1 - month * 0.02))
        credit_card_debt = max(0, credit_card_debt_base * (1 - month * 0.05))
        auto_loans = max(0, auto_loans_base * (1 - month * 0.03))
        other_debt = max(0, other_debt_base * (1 - month * 0.01))
        
        total_assets = cash_savings + investments + real_estate + retirement + other_assets
        total_liabilities = mortgage + student_loans + credit_card_debt + auto_loans + other_debt
        net_worth = total_assets - total_liabilities
        
        # Calculate changes
        if month > 1:
            prev_assets = history[-1]['total_assets']
            prev_liabilities = history[-1]['total_liabilities']
            assets_change = total_assets - prev_assets
            liabilities_change = total_liabilities - prev_liabilities
        else:
            assets_change = 0
            liabilities_change = 0
        
        history.append({
            'month': month,
            'cash_savings': cash_savings,
            'investments': investments,
            'real_estate': real_estate,
            'retirement': retirement,
            'other_assets': other_assets,
            'total_assets': total_assets,
            'mortgage': mortgage,
            'student_loans': student_loans,
            'credit_card_debt': credit_card_debt,
            'auto_loans': auto_loans,
            'other_debt': other_debt,
            'total_liabilities': total_liabilities,
            'net_worth': net_worth,
            'assets_change': assets_change,
            'liabilities_change': liabilities_change
        })
    
    return pd.DataFrame(history)


def calculate_net_worth_projection(
    current_net_worth: float,
    monthly_contribution: float,
    annual_return: float,
    years: int
) -> float:
    """Calculate projected net worth after given years."""
    
    monthly_return = (1 + annual_return) ** (1/12) - 1
    total_months = years * 12
    
    projected = current_net_worth
    for _ in range(total_months):
        projected = (projected + monthly_contribution) * (1 + monthly_return)
    
    return projected
