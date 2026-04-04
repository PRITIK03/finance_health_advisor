"""
Streamlit Web Application
Interactive dashboard for Financial Health Advisor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_full_dataset, calculate_financial_health_score
from preprocessing import FinancialDataPreprocessor, prepare_classification_data, calculate_fire_metrics, calculate_debt_paydown, calculate_emergency_fund_stress_test, calculate_subscription_audit
from models import train_all_models
from visualizations import FinancialVisualizer, generate_summary_statistics
from recommendations import RecommendationsEngine


@st.cache_data
def load_data():
    """Load or generate data."""
    users_df, monthly_df = generate_full_dataset(10000)
    return users_df, monthly_df


@st.cache_resource
def train_models(users_df, monthly_df):
    """Train ML models."""
    results, pipeline = train_all_models(users_df, monthly_df)
    return results, pipeline


def main():
    """Main application function."""
    
    st.set_page_config(
        page_title="Finance Health Advisor",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #1e293b !important;
    }
    
    .stApp {
        background-color: #f1f5f9 !important; /* Slightly more gray for better contrast with white cards */
    }

    /* Fix Metric visibility and layout */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 24px !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        border: 1px solid #e2e8f0 !important;
        transition: transform 0.2s ease-in-out;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        line-height: 1.2 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        margin-bottom: 8px !important;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    /* Container styling (using st.container(border=True)) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #f1f5f9 !important;
        margin-bottom: 24px !important;
        padding: 2rem !important;
    }

    /* Fix Plotly chart backgrounds */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }

    /* Fix Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding: 2rem 1rem !important;
    }

    [data-testid="stSidebarNav"] {
        background-color: transparent !important;
        padding-top: 1rem !important;
    }

    /* Main Content Padding */
    .block-container {
        padding: 3rem 4rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    /* Tab styling - more modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent !important;
        color: #64748b !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0 12px !important;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #2563eb !important;
    }

    /* Card Titles and Text */
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-text {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.6;
    }
    
    .highlight {
        color: #2563eb;
        font-weight: 600;
    }

    /* Hide redundant elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Ensure text inside markdown and headers is dark and visible */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
        line-height: 1.6 !important;
    }
    
    h1 { font-size: 2.25rem !important; font-weight: 800 !important; letter-spacing: -0.025em !important; }
    h2 { font-size: 1.5rem !important; font-weight: 700 !important; margin-top: 2rem !important; }
    
    /* Fix for Streamlit's info/warning boxes */
    [data-testid="stNotification"] {
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="stNotification"] p {
        color: inherit !important;
        font-weight: 500 !important;
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s !important;
    }

    .stDownloadButton>button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        width: 100% !important;
    }

    /* Dataframe rounding */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #e2e8f0 !important;
    }

    /* Micro-interactions */
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2) !important;
    }

    /* Glassmorphism Effect */
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"], [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stApp {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2845/2845812.png", width=80)
    with col2:
        st.title("Financial Health Advisor")
        st.markdown("<p style='font-size: 1.1rem; color: #64748b; margin-top: -15px;'>AI-Powered Financial Insights & Behavioral Analysis</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Load data
    with st.spinner("Generating synthetic financial data..."):
        users_df, monthly_df = load_data()
    
    # Train models
    with st.spinner("Training ML models..."):
        results, pipeline = train_models(users_df, monthly_df)
    
    # Preprocess data
    preprocessor = FinancialDataPreprocessor()
    users_processed = preprocessor.preprocess_users(users_df)
    monthly_processed = preprocessor.preprocess_monthly(monthly_df)
    
    # Visualizations
    visualizer = FinancialVisualizer(users_df, monthly_df)
    
    # Recommendations Engine
    recommendations_engine = RecommendationsEngine(users_df, monthly_df)

    # Sidebar
    with st.sidebar:
        st.markdown("<div style='text-align: center; padding-bottom: 20px;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2845/2845812.png", width=80)
        st.markdown("<h2 style='margin-top: 10px; color: #0f172a;'>Menu</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Theme Toggle
        theme = st.toggle("🌙 Dark Mode", value=False)
        
        if theme:
            st.markdown("""
            <style>
            .stApp { background-color: #0f172a !important; }
            [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"], [data-testid="stSidebar"] {
                background-color: #1e293b !important;
                border-color: #334155 !important;
                color: #f1f5f9 !important;
            }
            .card-title, h1, h2, h3, [data-testid="stMetricValue"] { color: #f8fafc !important; }
            .sidebar-text, [data-testid="stMetricLabel"], .stMarkdown p { color: #cbd5e1 !important; }
            .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
            .stTabs [aria-selected="true"] { color: #3b82f6 !important; }
            hr { border-color: #334155 !important; }
            </style>
            """, unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Go to section:",
            ["📊 Dashboard Overview", "🚨 Stress Test", "👥 Comparison Mode", "👥 User Segmentation", "🎯 Risk Prediction", 
             "📈 Forecasting", "🚨 Anomaly Detection", "💡 Recommendations", "🎯 Goal Planner", "🚀 Wealth Projection", "🔥 FIRE Tracker", "💸 Debt Optimizer", "👥 Peer Benchmarking", "🔮 Scenario Simulator", "🔍 Data Explorer"],
            label_visibility="collapsed"
        )
        
        # Sidebar Info
        st.markdown("---")
        st.subheader("About Advisor")
        st.markdown("""
        <div class='sidebar-text'>
        Uses <span class='highlight'>Advanced Machine Learning</span> to provide actionable financial intelligence.
        <br><br>
        <b>Key Modules:</b>
        <ul>
            <li>Predictive Risk Analysis</li>
            <li>Monte Carlo Simulations</li>
            <li>Peer Benchmarking</li>
            <li>Anomaly Pattern Detection</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown("---")
        st.subheader("System Insights")
        avg_health = users_df['financial_health_score'].mean()
        health_status = "Good" if avg_health > 70 else "Fair" if avg_health > 50 else "Poor"
        
        # Micro Bullet for Average Health vs Target
        st.plotly_chart(visualizer.create_mini_bullet(avg_health, 85, "Avg Health Index", color="#10b981"), use_container_width=True)
        
        # Financial Milestone Badges (New Feature)
        st.markdown("---")
        st.subheader("🏆 Achievements")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🛡️ **Debt-Free**")
            st.markdown("🔥 **FIRE Starter**")
        with col2:
            st.markdown("🌟 **Elite Credit**")
            st.markdown("🏦 **Wealth Builder**")
        
        # Top 3 Spending Categories Mini Bar
        st.markdown("---")
        spending_cols = ['Housing', 'Transportation', 'Food', 'Healthcare', 'Entertainment', 'Shopping', 'Education', 'Subscriptions', 'Insurance', 'Miscellaneous']
        top_spending = monthly_df[spending_cols].mean().sort_values(ascending=False).head(3)
        st.plotly_chart(visualizer.create_mini_bar(top_spending.index.tolist(), top_spending.values.tolist(), "Top 3 Avg Monthly Spend", color="#f59e0b"), use_container_width=True)
        
        st.markdown(f"""
        <div class='sidebar-text'>
        <b>Health Index:</b> <span class='highlight'>{avg_health:.1f} ({health_status})</span><br>
        <b>Model Status:</b> <span style='color: #10b981; font-weight: bold;'>Active</span><br>
        <b>User Base:</b> <span class='highlight'>{len(users_df):,}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Export Data Button
        st.markdown("---")
        st.subheader("Export Center")
        summary_data = users_df[['user_id', 'age', 'employment_type', 'monthly_income', 'monthly_expenses', 'financial_health_score', 'risk_label']].to_csv(index=False)
        st.download_button(
            label="📥 Export Analysis (CSV)",
            data=summary_data,
            file_name="financial_health_summary.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # ============ DASHBOARD OVERVIEW ============
    if page == "📊 Dashboard Overview":
        st.header("Financial Executive Summary")
        
        # Summary Statistics
        stats = generate_summary_statistics(users_df, monthly_df)
        
        # Metric cards with Gauge
        col1, col2 = st.columns([1.5, 2.5])
        
        with col1:
            with st.container(border=True):
                st.plotly_chart(visualizer.create_gauge_chart(stats['avg_health_score'], "System Health Index"), use_container_width=True)
                st.markdown(f"<div style='text-align: center; color: #64748b;'>Average score across <b>{stats['total_users']:,}</b> users</div>", unsafe_allow_html=True)

        with col2:
            sub_col1, sub_col2 = st.columns(2)
            
            # Prepare sparkline data
            monthly_income_trend = monthly_df.groupby('month')['income'].mean().tolist()
            monthly_savings_trend = monthly_df.groupby('month')['savings'].mean().tolist()
            
            with sub_col1:
                with st.container(border=False):
                    st.metric("Avg Monthly Income", f"${stats['avg_income']:,.0f}", 
                              help="The average gross monthly income across all user segments.")
                    st.plotly_chart(visualizer.create_sparkline(monthly_income_trend, color="#10b981"), use_container_width=True)
                
                with st.container(border=False):
                    st.metric("Avg Monthly Savings", f"${stats['avg_savings']:,.0f}", 
                              help="The average amount users are saving each month after expenses.")
                    st.plotly_chart(visualizer.create_sparkline(monthly_savings_trend, color="#3b82f6"), use_container_width=True)
                    
            with sub_col2:
                st.metric("Avg Credit Score", f"{stats['avg_credit_score']:.0f}", 
                          help="Average FICO score of the analyzed population.")
                st.metric("Data Sample Size", f"{stats['total_monthly_records']:,} mos", 
                          help="Total historical transaction months processed.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs for different views
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
            st.info("💡 **Subscription Leakage Detection:** AI analyzes your recurring costs to identify low-usage services.")
            selected_user_id = st.selectbox("Select User for Audit", users_df['user_id'].unique(), key="sub_audit_user")
            audit_df, audit_summary = calculate_subscription_audit(monthly_df, selected_user_id)

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Active Services", audit_summary['services_count'])
            with metric_col2:
                st.metric("Monthly Cost", f"${audit_summary['estimated_monthly_total']:,.0f}")
            with metric_col3:
                st.metric("Potential Savings", f"${audit_summary['potential_monthly_savings']:,.0f}/mo")
            with metric_col4:
                st.metric("High-Risk Leaks", audit_summary['high_risk_count'])

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
                    st.write(f"**Annual Subscription Spend:** `${audit_summary['estimated_annual_total']:,.0f}`")

            with st.container(border=True):
                st.markdown("<p class='card-title'>📋 Subscription Review Table</p>", unsafe_allow_html=True)
                st.dataframe(
                    audit_df[['name', 'cost', 'annual_cost', 'usage', 'recommendation', 'potential_monthly_savings']],
                    use_container_width=True
                )
                
        with tab4:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📋 Recent Financial Snapshot</p>", unsafe_allow_html=True)
                st.dataframe(users_df[['user_id', 'age', 'employment_type', 'monthly_income', 'monthly_expenses', 'financial_health_score']].head(20), use_container_width=True)
    
    # ============ STRESS TEST ============
    elif page == "🚨 Stress Test":
        st.header("Emergency Fund Stress Test")
        
        st.info("Model how your emergency fund performs during common financial shocks and compare it against the 6-month safety benchmark.")
        
        # User selector
        selected_user_id = st.selectbox(
            "Select User Profile for Stress Test",
            users_df['user_id'].unique(),
            format_func=lambda x: f"User {x}"
        )
        user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]

        with st.container(border=True):
            st.markdown("<p class='card-title'>⚙️ Stress Parameters</p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                crisis_type = st.radio(
                    "Select Crisis Scenario:",
                    ["Job Loss", "Medical Emergency", "Major Repair"],
                    horizontal=True
                )
            with col2:
                emergency_fund_months = st.slider(
                    "Emergency Fund Assumption (Months of Savings)",
                    1, 12, 6
                )

        stress_results = calculate_emergency_fund_stress_test(
            user_row,
            crisis_type=crisis_type,
            emergency_fund_months=emergency_fund_months
        )
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Emergency Fund (Estimated)", f"${stress_results['initial_fund']:,.0f}")
        with col2:
            st.metric("Survival Buffer", f"{stress_results['months_buffer']:.1f} months")
        with col3:
            st.metric(
                "Fund Gap",
                f"${stress_results['fund_gap']:,.0f}",
                delta="On target" if stress_results['fund_gap'] == 0 else "Below target",
                delta_color="inverse"
            )
        with col4:
            st.metric(
                "Safety Band",
                stress_results['safety_band'],
                delta=f"{stress_results['months_buffer'] - 6:.1f} mo vs benchmark",
                delta_color="normal"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.5, 2.5])
        
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🛟 Resilience Snapshot</p>", unsafe_allow_html=True)
                st.plotly_chart(
                    visualizer.create_gauge_chart(
                        min(100, (stress_results['months_buffer'] / 6) * 100),
                        "Emergency Readiness",
                        target=100
                    ),
                    use_container_width=True
                )
                st.write(f"**Scenario:** {crisis_type}")
                st.info(stress_results['impact_message'])
                st.write(f"**One-Time Shock:** `${stress_results['one_time_cost']:,.0f}`")
                st.write(f"**Monthly Crisis Expenses:** `${stress_results['crisis_expenses']:,.0f}`")
                st.write(f"**Available Fund After Shock:** `${stress_results['remaining_fund_start']:,.0f}`")
                st.write(f"**Recommended 6-Month Buffer:** `${stress_results['recommended_fund']:,.0f}`")
        
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📅 Survival Timeline</p>", unsafe_allow_html=True)
                st.plotly_chart(
                    visualizer.create_emergency_fund_chart(stress_results),
                    use_container_width=True
                )

        with st.container(border=True):
            st.markdown("<p class='card-title'>💡 Action Plan</p>", unsafe_allow_html=True)
            if stress_results['is_safe']:
                st.success("Your current buffer clears the 6-month benchmark for this scenario.")
            elif stress_results['months_buffer'] >= 3:
                st.warning("You have some protection, but the buffer is thinner than the recommended emergency reserve.")
            else:
                st.error("This scenario drains your fund quickly. Building liquidity should be a near-term priority.")

            monthly_gap_savings = stress_results['fund_gap'] / 12 if stress_results['fund_gap'] > 0 else 0
            st.write(
                f"Closing the gap over 12 months would require about **${monthly_gap_savings:,.0f}/month** in extra reserve savings."
            )

    # ============ COMPARISON MODE ============
    elif page == "👥 Comparison Mode":
        st.header("Side-by-Side User Comparison")
        
        st.info("Compare two financial profiles side-by-side to identify relative strengths and opportunities.")
        
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
        else:
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

    # ============ USER SEGMENTATION ============
    elif page == "👥 User Segmentation":
        st.header("User Segmentation Analysis")
        
        st.info("K-Means clustering identifies 5 distinct financial profiles based on behavior and wealth metrics.")
        
        tab1, tab2 = st.tabs(["📊 Cluster Overview", "📋 Detailed Statistics"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>🎯 K-Means Cluster Profiles</p>", unsafe_allow_html=True)
                    st.plotly_chart(visualizer.create_cluster_analysis_plot(), use_container_width=True)
                
            with col2:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>👥 Distribution</p>", unsafe_allow_html=True)
                    cluster_counts = users_df['cluster'].value_counts().sort_index()
                    for i, count in cluster_counts.items():
                        st.markdown(f"**Cluster {i}:** `{count}` users")
        
        with tab2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📋 Cluster Financial Averages</p>", unsafe_allow_html=True)
                cluster_stats = users_df.groupby('cluster').agg({
                    'monthly_income': 'mean',
                    'monthly_expenses': 'mean',
                    'monthly_savings': 'mean',
                    'credit_score': 'mean',
                    'financial_health_score': 'mean'
                }).round(2)
                st.dataframe(cluster_stats, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Profile Descriptions")
            cols = st.columns(3)
            for i, cluster_id in enumerate(sorted(users_df['cluster'].unique())):
                with cols[i % 3]:
                    cluster_data = users_df[users_df['cluster'] == cluster_id]
                    with st.container(border=True):
                        st.markdown(f"<p class='card-title'>Profile {cluster_id}</p>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class='sidebar-text'>
                        Avg Health: <span class='highlight'>{cluster_data['financial_health_score'].mean():.1f}</span><br>
                        Avg Income: <b>${cluster_data['monthly_income'].mean():,.0f}</b><br>
                        Users: {len(cluster_data)}
                        </div>
                        """, unsafe_allow_html=True)
    
    # ============ RISK PREDICTION ============
    elif page == "🎯 Risk Prediction":
        st.header("Financial Risk Analysis")
        
        st.info("Random Forest classifier predicts financial vulnerability by analyzing user profiles and historical data.")
        
        # Metric cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", f"{results['classification']['train_metrics']['accuracy']:.1%}")
        with col2:
            st.metric("Test Accuracy", f"{results['classification']['test_metrics']['accuracy']:.1%}")
        with col3:
            st.metric("Model Reliability", f"{results['classification']['test_metrics']['cv_mean']:.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🎯 Prediction Insights", "📊 Importance & Distribution"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>🎯 Risk Category Distribution</p>", unsafe_allow_html=True)
                    risk_counts = users_df['risk_label'].value_counts()
                    colors = {'Very Low': '#10b981', 'Low': '#34d399', 'Medium': '#f59e0b', 
                             'High': '#f97316', 'Very High': '#ef4444'}
                    
                    fig = px.pie(
                        values=risk_counts.values, 
                        names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map=colors,
                        hole=0.6
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
                    st.markdown("<p class='card-title'>🧠 Risk Intelligence</p>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='sidebar-text'>
                    The AI model classifies users into 5 risk levels by analyzing <b>{len(users_df.columns)}</b> financial dimensions.
                    <br><br>
                    <b>Top Risk Factors:</b>
                    <ul>
                        <li><span class='highlight'>Credit Score</span>: Impact on borrowing capacity</li>
                        <li><span class='highlight'>Monthly Savings</span>: Buffer against volatility</li>
                        <li><span class='highlight'>Debt-to-Income</span>: Leverage sustainability</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
        with tab2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📊 Model Feature Importance</p>", unsafe_allow_html=True)
                X_class, y_class, le = prepare_classification_data(users_processed)
                
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_class.values, y_class)
                
                importance = pd.DataFrame({
                    'feature': X_class.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(
                    importance, 
                    x='importance', 
                    y='feature', 
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    template="plotly_white", 
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ============ FORECASTING ============
    elif page == "📈 Forecasting":
        st.header("Savings Forecasting")
        
        st.info("Gradient Boosting regressor predicts future savings potential based on spending habits and income stability.")
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Variance (R²)", f"{results['forecasting']['test_metrics']['r2']:.2f}")
        with col2:
            st.metric("Avg Prediction Error", f"${results['forecasting']['test_metrics']['mae']:.0f}")
        with col3:
            st.metric("RMSE", f"${results['forecasting']['test_metrics']['rmse']:.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📈 Trends", "📊 Segment Analysis"])
        
        with tab1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📈 Monthly Savings Trends & Projection</p>", unsafe_allow_html=True)
                
                monthly_avg = monthly_df.groupby('month').agg({
                    'income': 'mean',
                    'expenses': 'mean',
                    'savings': 'mean'
                }).reset_index()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=monthly_avg['month'],
                    y=monthly_avg['savings'],
                    mode='lines+markers',
                    name='Actual Savings',
                    line=dict(color='#3b82f6', width=4, shape='spline'),
                    marker=dict(size=10, borderwidth=2, color='white', line=dict(color='#3b82f6'))
                ))
                
                # Add trend line
                z = np.polyfit(monthly_avg['month'], monthly_avg['savings'], 2)
                p = np.poly1d(z)
                x_trend = np.arange(1, 13)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='AI Forecast Trend',
                    line=dict(color='#f43f5e', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title='Month of Year',
                    yaxis_title='Average Savings ($)',
                    xaxis=dict(tickmode='linear', tick0=1, dtick=1, gridcolor='#f1f5f9'),
                    yaxis=dict(gridcolor='#f1f5f9'),
                    margin=dict(t=30, b=30, l=30, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📊 Savings Performance by Risk Profile</p>", unsafe_allow_html=True)
                
                savings_by_risk = monthly_df.groupby('risk_label')['savings'].mean().sort_values()
                
                fig = px.bar(
                    x=savings_by_risk.index,
                    y=savings_by_risk.values,
                    labels={'x': 'Risk Category', 'y': 'Average Savings ($)'},
                    color=savings_by_risk.values,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(
                    template="plotly_white", 
                    margin=dict(t=30, b=30, l=30, r=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ============ ANOMALY DETECTION ============
    elif page == "🚨 Anomaly Detection":
        st.header("Anomaly Detection")
        
        st.info("Isolation Forest algorithm identifies outliers in spending and savings that deviate from established user patterns.")
        
        # Anomaly statistics
        anomaly_count = monthly_df['ml_anomaly'].sum()
        anomaly_pct = anomaly_count / len(monthly_df) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Anomalies Detected", f"{anomaly_count}", delta=None)
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_pct:.2f}%", delta=None, delta_color="inverse")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🚨 Visual Analysis", "📋 Detected Anomalies"])
        
        with tab1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🚨 Spending & Savings Anomalies</p>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_anomaly_analysis_plot(), use_container_width=True)
            
        with tab2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🔍 Flagged Transactions (AI Insights)</p>", unsafe_allow_html=True)
                
                anomalies = monthly_df[monthly_df['ml_anomaly'] == 1]
                st.dataframe(
                    anomalies[['user_id', 'month', 'income', 'expenses', 'savings', 'Entertainment', 'Shopping']].head(20),
                    use_container_width=True
                )
    
    # ============ RECOMMENDATIONS ============
    elif page == "💡 Recommendations":
        st.header("Financial Recommendations")
        
        st.info("AI-powered recommendations tailored to your financial profile based on behavioral ML analysis.")
        
        # Recommendation type selector
        rec_type = st.radio(
            "Select Analysis Level:",
            ["👤 Individual User", "👥 Cohort Analysis"],
            horizontal=True
        )
        
        if rec_type == "👤 Individual User":
            # User selector
            user_id = st.selectbox(
                "Select a User Profile",
                users_df['user_id'].unique(),
                format_func=lambda x: f"User {x}"
            )
            
            # Get user data
            user_row = users_df[users_df['user_id'] == user_id].iloc[0]
            
            # Display user summary
            col1, col2 = st.columns([1.5, 2.5])
            
            with col1:
                with st.container(border=True):
                    st.plotly_chart(visualizer.create_gauge_chart(user_row['financial_health_score'], "Health Index"), use_container_width=True)
                    st.markdown(f"<div style='text-align: center; font-weight: 700; color: #2563eb;'>{user_row['risk_label']} Risk</div>", unsafe_allow_html=True)

            with col2:
                sub_col1, sub_col2 = st.columns(2)
                
                # User's personal trends for sparklines
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
            
            # Get recommendations
            user_recs = recommendations_engine.generate_user_recommendations(user_id)
            diversification_profile = recommendations_engine.get_diversification_profile(user_row.to_dict())
            
            # Display recommendations in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["💰 Budget", "💳 Debt", "🏦 Savings", "📈 Investments"])
            
            with tab1:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>📝 Budget Recommendations</p>", unsafe_allow_html=True)
                    for rec in user_recs['budget']:
                        status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠'}.get(rec['status'], '⚪')
                        st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                        st.markdown(f"   *💡 {rec['suggestion']}*")
                        st.markdown("")
            
            with tab2:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>💳 Debt Management</p>", unsafe_allow_html=True)
                    for rec in user_recs['debt']:
                        status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'excellent': '💚', 'info': '🔵'}.get(rec['status'], '⚪')
                        st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                        st.markdown(f"   *💡 {rec['suggestion']}*")
                        st.markdown("")
            
            with tab3:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>🏦 Savings & Safety Net</p>", unsafe_allow_html=True)
                    for rec in user_recs['savings']:
                        status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'info': '🔵'}.get(rec['status'], '⚪')
                        st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                        st.markdown(f"   *💡 {rec['suggestion']}*")
                        st.markdown("")
            
            with tab4:
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
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'info': '🔵'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")

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
            # Cohort Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_filter = st.selectbox(
                    "Filter by Cluster Profile",
                    ['All'] + list(range(5))
                )
            with col2:
                risk_filter_cohort = st.selectbox(
                    "Filter by Risk Level",
                    ['All', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
                )
            
            cluster_val = None if cluster_filter == 'All' else cluster_filter
            risk_val = None if risk_filter_cohort == 'All' else risk_filter_cohort
            
            # Get cohort recommendations
            cohort_recs = recommendations_engine.generate_cohort_recommendations(cluster_val, risk_val)
            
            if cohort_recs['cohort_size'] > 0:
                # Cohort statistics
                stats = cohort_recs['cohort_stats']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cohort Size", f"{cohort_recs['cohort_size']:,}")
                with col2:
                    st.metric("Avg Income", f"${stats['avg_income']:,.0f}")
                with col3:
                    st.metric("Avg Savings", f"${stats['avg_savings']:,.0f}")
                with col4:
                    st.metric("Avg Health Score", f"{stats['avg_health_score']:.1f}/100")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"✨ **Cohort Summary:** {cohort_recs['summary']}")
                
                # Display cohort recommendations
                tab1, tab2, tab3, tab4 = st.tabs(["💰 Budget", "💳 Debt", "🏦 Savings", "📈 Investments"])
                
                all_recs = cohort_recs['recommendations']
                
                with tab1:
                    with st.container(border=True):
                        st.markdown("<p class='card-title'>📝 Budget Recommendations</p>", unsafe_allow_html=True)
                        for rec in all_recs['budget']:
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")
                
                with tab2:
                    with st.container(border=True):
                        st.markdown("<p class='card-title'>💳 Debt Management</p>", unsafe_allow_html=True)
                        for rec in all_recs['debt']:
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'excellent': '💚', 'info': '🔵'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")
                
                with tab3:
                    with st.container(border=True):
                        st.markdown("<p class='card-title'>🏦 Savings & Safety Net</p>", unsafe_allow_html=True)
                        for rec in all_recs['savings']:
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'info': '🔵'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")
                
                with tab4:
                    with st.container(border=True):
                        st.markdown("<p class='card-title'>📈 Investment Strategy</p>", unsafe_allow_html=True)
                        for rec in all_recs['investments']:
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'info': '🔵'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")
            else:
                st.warning("No users found in the selected cohort.")

    # ============ GOAL PLANNER ============
    elif page == "🎯 Goal Planner":
        st.header("Financial Goal Roadmap")
        
        st.info("Plan your financial milestones and see when you'll reach them based on your current savings profile.")
        
        # User selector for profile baseline
        selected_user_id = st.selectbox(
            "Select a User Profile for Planning",
            users_df['user_id'].unique(),
            format_func=lambda x: f"User {x}"
        )
        
        user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]
        
        # Goal inputs
        with st.container(border=True):
            st.markdown("<p class='card-title'>🎯 Define Your Financial Goal</p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                goal_type = st.selectbox("Goal Type", ["Emergency Fund", "Car", "Home Downpayment", "Travel", "Custom"])
                target_amount = st.number_input("Target Amount ($)", 1000, 1000000, 10000)
            
            with col2:
                current_savings_rate = user_row['monthly_savings']
                extra_monthly = st.slider("Additional Monthly Savings ($)", 0, 5000, 0)
                total_monthly_savings = current_savings_rate + extra_monthly

        # Calculation
        months_to_reach = target_amount / total_monthly_savings if total_monthly_savings > 0 else float('inf')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Roadmap
        if months_to_reach != float('inf'):
            years = int(months_to_reach // 12)
            months = int(months_to_reach % 12)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Monthly Savings", f"${total_monthly_savings:,.0f}", delta=f"${extra_monthly:,.0f}")
            with col2:
                st.metric("Time to Reach Goal", f"{years}y {months}m")
            with col3:
                # Calculate optimization
                boost_savings = total_monthly_savings * 1.2
                boost_months = target_amount / boost_savings
                time_saved = months_to_reach - boost_months
                st.metric("Time Saved (20% Boost)", f"{int(time_saved)} mo", delta="Optimized", delta_color="normal")

            # Visual Roadmap
            with st.container(border=True):
                st.markdown("<p class='card-title'>📅 Achievement Timeline</p>", unsafe_allow_html=True)
                
                timeline_months = list(range(1, int(min(months_to_reach + 12, 120)) + 1))
                savings_over_time = [total_monthly_savings * m for m in timeline_months]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timeline_months, 
                    y=savings_over_time, 
                    fill='tozeroy', 
                    name='Savings Growth', 
                    line=dict(color='#3b82f6', width=3),
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                fig.add_hline(y=target_amount, line_dash="dash", line_color="#ef4444", 
                             annotation_text=f"Goal: ${target_amount:,.0f}", annotation_position="top left")
                
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Months from Now",
                    yaxis_title="Total Savings ($)",
                    margin=dict(t=30, b=30, l=30, r=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Actionable Roadmap
            st.success(f"### 🏆 Success Roadmap for your {goal_type}")
            st.write(f"Based on your current savings rate of **${total_monthly_savings:,.0f}/month**, you will reach your **${target_amount:,.0f}** goal in **{years} years and {months} months**.")
            
            st.info(f"💡 **Pro Tip:** If you reduce your discretionary spending by just 15%, you could reach your goal **{int(months_to_reach * 0.15)} months earlier**.")
        else:
            st.error("Please ensure your monthly savings is greater than 0 to reach your goal.")

    # ============ WEALTH PROJECTION ============
    elif page == "🚀 Wealth Projection":
        st.header("Wealth & Retirement Projection")
        
        st.info("Monte Carlo simulation projects your future wealth based on savings, investments, and market volatility.")
        
        # User selector
        selected_user_id = st.selectbox(
            "Select User Profile",
            users_df['user_id'].unique(),
            format_func=lambda x: f"User {x}"
        )
        user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]
        
        # Simulation Parameters
        with st.container(border=True):
            st.markdown("<p class='card-title'>⚙️ Simulation Settings</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                projection_years = st.slider("Projection Horizon (Years)", 5, 50, 30)
                initial_wealth = st.number_input("Current Assets ($)", 0, 5000000, int(user_row['monthly_savings'] * 24))
            
            with col2:
                avg_return = st.slider("Expected Annual Return (%)", 1.0, 15.0, 8.0) / 100
                volatility = st.slider("Market Volatility (%)", 5.0, 40.0, 18.0) / 100
                
            with col3:
                monthly_contribution = st.number_input("Monthly Contribution ($)", 0, 50000, int(user_row['monthly_savings']))
                num_simulations = st.selectbox("Number of Simulations", [100, 500, 1000], index=1)
                
            # Inflation Toggle (New Feature)
            use_inflation = st.toggle("Adjust for Inflation", value=True)
            inflation_rate = st.slider("Expected Annual Inflation (%)", 0.0, 10.0, 3.0) / 100 if use_inflation else 0
            
        # Monte Carlo Simulation Logic
        # Adjust return for inflation
        real_avg_return = avg_return - inflation_rate
        results_mc = np.zeros((num_simulations, projection_years + 1))
        results_mc[:, 0] = initial_wealth
        
        for i in range(num_simulations):
            current_wealth = initial_wealth
            for year in range(1, projection_years + 1):
                yearly_return = np.random.normal(real_avg_return, volatility)
                current_wealth = (current_wealth + monthly_contribution * 12) * (1 + yearly_return)
                results_mc[i, year] = max(0, current_wealth)
        
        # Results Display
        st.markdown("<br>", unsafe_allow_html=True)
        median_final = np.median(results_mc[:, -1])
        p10_final = np.percentile(results_mc[:, -1], 10)
        p90_final = np.percentile(results_mc[:, -1], 90)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Median Wealth ({projection_years}y)", f"${median_final:,.0f}")
        with col2:
            st.metric("Pessimistic (10%)", f"${p10_final:,.0f}")
        with col3:
            st.metric("Optimistic (90%)", f"${p90_final:,.0f}")
            
        # Chart
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Probabilistic Wealth Growth</p>", unsafe_allow_html=True)
            st.plotly_chart(visualizer.create_wealth_projection_chart(results_mc, projection_years), use_container_width=True)
            
        # Insights
        with st.container(border=True):
            st.markdown("<p class='card-title'>🏖️ Retirement Readiness</p>", unsafe_allow_html=True)
            
            annual_withdrawal = median_final * 0.04
            monthly_income_potential = annual_withdrawal / 12
            
            st.write(f"Based on the median projection, your wealth could generate approximately **${monthly_income_potential:,.0f}/month** in retirement (using the 4% rule).")
            
            if monthly_income_potential < user_row['monthly_expenses']:
                st.warning(f"⚠️ **Caution:** Your projected retirement income is less than your current expenses (${user_row['monthly_expenses']:,.0f}). Consider increasing your monthly contributions.")
            else:
                st.success(f"✅ **Great Job!** Your projected retirement income exceeds your current expenses. You are on a solid path to financial independence.")

    # ============ FIRE TRACKER ============
    elif page == "🔥 FIRE Tracker":
        st.header("Financial Independence (FIRE) Tracker")
        
        st.info("Calculate your FIRE Number and track your journey toward financial freedom using the 4% Rule.")
        
        # User selector
        selected_user_id = st.selectbox(
            "Select User Profile",
            users_df['user_id'].unique(),
            format_func=lambda x: f"User {x}"
        )
        user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]
        
        # FIRE Settings
        with st.container(border=True):
            st.markdown("<p class='card-title'>⚙️ FIRE Parameters</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                swr = st.slider("Safe Withdrawal Rate (%)", 2.0, 5.0, 4.0, 0.1) / 100
                use_fire_inflation = st.toggle("Adjust for Inflation (FIRE)", value=True, key="fire_infl_toggle")
            with col2:
                custom_expenses = st.number_input("Target Annual Expenses ($)", 0, 500000, int(user_row['monthly_expenses'] * 12))
                fire_inflation_rate = st.slider("Inflation Rate (%)", 0.0, 10.0, 3.0, key="fire_infl_rate") / 100 if use_fire_inflation else 0
            with col3:
                current_stash = st.number_input("Current Investable Assets ($)", 0, 10000000, int(user_row['monthly_investments'] * 12 * 5))
        
        # Override user row for calculations
        calc_row = user_row.copy()
        calc_row['monthly_expenses'] = custom_expenses / 12
        calc_row['monthly_investments'] = current_stash / 60 # Back-calculate monthly contribution for simplicity
        
        fire_metrics = calculate_fire_metrics(calc_row, safe_withdrawal_rate=swr, inflation_rate=fire_inflation_rate)
        # Fix the current investments to the user's input
        fire_metrics['current_investments'] = current_stash
        fire_metrics['progress_pct'] = min(100, (current_stash / fire_metrics['fire_number']) * 100) if fire_metrics['fire_number'] > 0 else 0
        
        # Main Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("FIRE Number", f"${fire_metrics['fire_number']:,.0f}")
        with col2:
            st.metric("Current Progress", f"{fire_metrics['progress_pct']:.1f}%")
        with col3:
            years = fire_metrics['years_to_fire']
            st.metric("Years to Freedom", f"{years if years < 100 else '∞'} years")
            
        # Visualization
        tab1, tab2 = st.tabs(["📊 Progress Overview", "📈 FIRE Roadmap"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            with col1:
                with st.container(border=True):
                    st.plotly_chart(visualizer.create_gauge_chart(fire_metrics['progress_pct'], "FIRE Progress Index"), use_container_width=True)
            with col2:
                with st.container(border=True):
                    st.plotly_chart(visualizer.create_fire_progress_chart(fire_metrics), use_container_width=True)
                    
        with tab2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🗺️ Journey to Independence</p>", unsafe_allow_html=True)
                
                # Projection for roadmap
                years_proj = min(50, int(fire_metrics['years_to_fire'] + 10))
                monthly_contrib = user_row['monthly_savings'] + user_row['monthly_investments']
                
                x_vals = list(range(years_proj + 1))
                y_vals = []
                wealth = current_stash
                for y in x_vals:
                    y_vals.append(wealth)
                    wealth = (wealth + monthly_contrib * 12) * 1.07
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', name='Projected Stash', line_color='#2563eb'))
                fig.add_hline(y=fire_metrics['fire_number'], line_dash="dash", line_color="#10b981", annotation_text="FIRE Target")
                fig.add_hline(y=fire_metrics['lean_fire_number'], line_dash="dot", line_color="#f59e0b", annotation_text="LeanFIRE Target")
                
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Years from Now",
                    yaxis_title="Portfolio Value ($)",
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ============ DEBT OPTIMIZER ============
    elif page == "💸 Debt Optimizer":
        st.header("Debt Paydown Optimizer")
        
        st.info("Optimize your debt repayment strategy by comparing the Avalanche (interest-focused) and Snowball (balance-focused) methods.")
        
        # User selector
        selected_user_id = st.selectbox(
            "Select User Profile for Analysis",
            users_df['user_id'].unique(),
            format_func=lambda x: f"User {x}"
        )
        user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]
        
        # Debt setup
        with st.container(border=True):
            st.markdown("<p class='card-title'>💳 Your Debt Portfolio</p>", unsafe_allow_html=True)
            
            # Default debt split for simulation based on user's total_debt
            total_debt = user_row['total_debt']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Total Debt to Re-allocate:** `${total_debt:,.0f}`")
                st.markdown(f"**Current Monthly Payment:** `${user_row['monthly_loan_payments']:,.0f}`")
            
            with col2:
                extra_payment = st.slider("Additional Monthly Payment ($)", 0, 5000, 200)
            
            st.markdown("---")
            st.write("Customize your debts for a more accurate simulation:")
            
            # Simplified debt inputs
            debt_col1, debt_col2, debt_col3 = st.columns(3)
            
            # Split total debt into 3 hypothetical accounts
            with debt_col1:
                d1_bal = st.number_input("Debt 1 Balance ($)", 0.0, total_debt, total_debt * 0.2)
                d1_rate = st.number_input("Debt 1 Interest (%)", 0.0, 35.0, 18.0)
            with debt_col2:
                d2_bal = st.number_input("Debt 2 Balance ($)", 0.0, total_debt, total_debt * 0.3)
                d2_rate = st.number_input("Debt 2 Interest (%)", 0.0, 35.0, 12.0)
            with debt_col3:
                d3_bal = st.number_input("Debt 3 Balance ($)", 0.0, total_debt, total_debt * 0.5)
                d3_rate = st.number_input("Debt 3 Interest (%)", 0.0, 35.0, 7.0)
            
            debts = [
                {'name': 'Credit Card', 'balance': d1_bal, 'interest_rate': d1_rate, 'min_payment': d1_bal * 0.02},
                {'name': 'Personal Loan', 'balance': d2_bal, 'interest_rate': d2_rate, 'min_payment': d2_bal * 0.03},
                {'name': 'Auto/Other', 'balance': d3_bal, 'interest_rate': d3_rate, 'min_payment': d3_bal * 0.01}
            ]
            
            # Validate total balance
            current_total = sum(d['balance'] for d in debts)
            if abs(current_total - total_debt) > 1:
                st.warning(f"Note: Your customized total balance (${current_total:,.0f}) differs from the user profile total (${total_debt:,.0f}).")

        # Run Simulation
        debt_results = calculate_debt_paydown(debts, extra_payment)
        
        # Display Results
        col1, col2, col3 = st.columns(3)
        
        best_strategy = "Avalanche" if debt_results['avalanche']['total_interest'] < debt_results['snowball']['total_interest'] else "Snowball"
        interest_saved = abs(debt_results['avalanche']['total_interest'] - debt_results['snowball']['total_interest'])
        
        with col1:
            st.metric("Total Interest (Avalanche)", f"${debt_results['avalanche']['total_interest']:,.0f}")
        with col2:
            st.metric("Total Interest (Snowball)", f"${debt_results['snowball']['total_interest']:,.0f}")
        with col3:
            st.metric("Optimal Strategy", best_strategy, delta=f"${interest_saved:,.0f} saved", delta_color="normal")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📈 Paydown Timeline", "📊 Strategy Comparison"])
        
        with tab1:
            with st.container(border=True):
                st.plotly_chart(visualizer.create_debt_paydown_chart(debt_results), use_container_width=True)
        
        with tab2:
            with st.container(border=True):
                st.plotly_chart(visualizer.create_interest_savings_chart(debt_results), use_container_width=True)
                
        # Insights
        with st.container(border=True):
            st.markdown("<p class='card-title'>💡 Expert Insights</p>", unsafe_allow_html=True)
            
            if best_strategy == "Avalanche":
                st.success(f"### 🚀 Go with the Avalanche Method")
                st.write(f"By focusing on your highest interest rate debt first, you will save **${interest_saved:,.0f}** in interest compared to the Snowball method.")
            else:
                st.info(f"### ❄️ Go with the Snowball Method")
                st.write(f"In this scenario, the Snowball method is slightly more efficient or comparable, and provides quick wins by clearing small balances first.")
                
            st.markdown(f"**Debt-Free Date:** You will be debt-free in approximately **{debt_results[best_strategy.lower()]['months']} months** using the {best_strategy} strategy with your extra **${extra_payment:,.0f}/month** payment.")
            
            if extra_payment == 0:
                st.warning("⚠️ **Tip:** Even an extra $100 per month could drastically reduce your total interest and time to debt-free.")

    # ============ PEER BENCHMARKING ============
    elif page == "👥 Peer Benchmarking":
        st.header("Peer Benchmarking Analysis")
        
        st.info("Compare your financial profile with similar users in our database to see where you stand.")
        
        # User selector
        selected_user_id = st.selectbox(
            "Select a User to Analyze",
            users_df['user_id'].unique(),
            format_func=lambda x: f"User {x}"
        )
        
        user_idx = users_df[users_df['user_id'] == selected_user_id].index[0]
        user_data = users_df.iloc[user_idx]
        
        # Find similar users using KNN
        from preprocessing import prepare_clustering_data
        clustering_data = prepare_clustering_data(users_df)
        
        # Get integer position for KNN search
        user_pos = users_df.index.get_loc(user_idx)
        
        similar_indices = pipeline.similar_users_model.find_similar(clustering_data.values, user_pos, n=5)
        similar_users = users_df.iloc[similar_indices]
        
        # Peer comparison metrics
        peers_avg = similar_users.mean(numeric_only=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🕸️ Peer Comparison Radar</p>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_peer_comparison_radar(user_data, peers_avg), use_container_width=True)
        
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📊 Benchmarking Insights</p>", unsafe_allow_html=True)
                
                # Comparison items
                comp_items = [
                    ('Monthly Income', 'monthly_income', '$'),
                    ('Monthly Savings', 'monthly_savings', '$'),
                    ('Health Score', 'financial_health_score', ''),
                    ('Credit Score', 'credit_score', '')
                ]
                
                for label, col, unit in comp_items:
                    u_val = user_data[col]
                    p_val = peers_avg[col]
                    diff = (u_val - p_val) / p_val * 100 if p_val != 0 else 0
                    
                    color = "#10b981" if u_val >= p_val else "#ef4444"
                    icon = "📈" if u_val >= p_val else "📉"
                    
                    st.markdown(f"""
                    <div style='margin-bottom: 15px;'>
                        <div style='font-weight: 600; font-size: 0.9rem; color: #64748b;'>{label}</div>
                        <div style='display: flex; align-items: baseline; gap: 10px;'>
                            <span style='font-size: 1.2rem; font-weight: 700; color: #0f172a;'>{unit}{u_val:,.1f}</span>
                            <span style='font-size: 0.85rem; color: #94a3b8;'>vs Peer Avg {unit}{p_val:,.1f}</span>
                            <span style='color: {color}; font-weight: bold; font-size: 0.85rem;'>({icon} {diff:+.1f}%)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"**Financial Neighbors Found**: `{len(similar_users)}` users")
                st.dataframe(similar_users[['user_id', 'age', 'employment_type', 'financial_health_score', 'risk_label']], use_container_width=True)

    # ============ SCENARIO SIMULATOR ============
    elif page == "🔮 Scenario Simulator":
        st.header("Financial Scenario Simulator")
        
        st.info("Simulate how changes in your income, expenses, and debt would affect your overall financial health score and risk category.")
        
        # Simulator inputs
        with st.container(border=True):
            st.markdown("<p class='card-title'>🔮 Adjust Your Financial Parameters</p>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                sim_income = st.slider("Monthly Income ($)", 0, 50000, 5000)
                sim_expenses = st.slider("Monthly Expenses ($)", 0, 50000, 3500)
            with col2:
                sim_savings = st.slider("Monthly Savings ($)", 0, 50000, 1000)
                sim_investments = st.slider("Monthly Investments ($)", 0, 50000, 500)
            with col3:
                sim_credit = st.slider("Credit Score", 300, 850, 700)
                sim_debt = st.slider("Total Debt ($)", 0, 1000000, 50000)
        
        # Calculation logic
        sim_profile = {
            'monthly_income': sim_income,
            'monthly_expenses': sim_expenses,
            'monthly_savings': sim_savings,
            'monthly_investments': sim_investments,
            'credit_score': sim_credit,
            'total_debt': sim_debt
        }
        
        sim_health_score = calculate_financial_health_score(sim_profile)
        
        # Prepare for ML prediction
        sim_features = pd.DataFrame([{
            'age': 35,
            'monthly_income': sim_income,
            'monthly_expenses': sim_expenses,
            'monthly_savings': sim_savings,
            'monthly_investments': sim_investments,
            'credit_score': sim_credit,
            'total_debt': sim_debt,
            'monthly_loan_payments': sim_income * 0.1,  # Assumption
            'financial_health_score': sim_health_score,
            'employment_type_encoded': 0  # Assumption
        }])
        
        sim_features['savings_rate'] = sim_features['monthly_savings'] / sim_features['monthly_income'].replace(0, 1)
        sim_features['expense_ratio'] = sim_features['monthly_expenses'] / sim_features['monthly_income'].replace(0, 1)
        sim_features['investment_rate'] = sim_features['monthly_investments'] / sim_features['monthly_income'].replace(0, 1)
        sim_features['debt_to_income'] = sim_features['total_debt'] / (sim_features['monthly_income'] * 12).replace(0, 1)
        sim_features['loan_to_income'] = sim_features['monthly_loan_payments'] / sim_features['monthly_income'].replace(0, 1)
        
        # Ensure correct column order for classifier
        sim_features = sim_features[pipeline.classification_model.feature_names]
        
        # Predict risk
        sim_risk_idx = pipeline.classification_model.predict(sim_features)[0]
        _, _, le = prepare_classification_data(users_df)
        sim_risk_label = le.inverse_transform([sim_risk_idx])[0]
        
        # Simulation Results
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.5, 2.5])
        
        with col1:
            with st.container(border=True):
                st.plotly_chart(visualizer.create_gauge_chart(sim_health_score, "Simulated Health Index"), use_container_width=True)
                st.markdown(f"<div style='text-align: center; font-weight: 700; color: #2563eb;'>{sim_risk_label} Risk Category</div>", unsafe_allow_html=True)
                
        with col2:
            sub_col1, sub_col2 = st.columns(2)
            avg_income = users_df['monthly_income'].mean()
            avg_health = users_df['financial_health_score'].mean()
            income_delta = (sim_income - avg_income) / avg_income * 100
            health_delta = sim_health_score - avg_health
            
            with sub_col1:
                st.metric("Income vs Market", f"${sim_income:,.0f}", f"{income_delta:+.1f}%")
            with sub_col2:
                st.metric("Health vs Market", f"{sim_health_score:.1f}", f"{health_delta:+.1f} pts")
            
            with st.container(border=True):
                st.markdown(f"**Savings Rate:** {sim_income > 0 and sim_savings/sim_income:.1%}")
                st.markdown(f"**Debt/Income:** {sim_income > 0 and sim_debt/(sim_income*12):.1%}")
        
        # Actionable insights
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p class='card-title'>💡 Simulation Insights</p>", unsafe_allow_html=True)
            if sim_health_score < 50:
                st.warning("🚨 **Caution:** Your simulated profile shows high financial stress. Consider reducing debt and increasing the savings rate.")
            elif sim_health_score < 75:
                st.info("✨ **Solid Foundation:** You have a solid foundation. To reach an excellent score, focus on increasing investments and maintaining a high credit score.")
            else:
                st.success("🏆 **Excellent Score:** Excellent! This simulated profile represents strong financial health. Keep up the disciplined savings approach.")
            st.write("**Pro Tip:** Try increasing your 'Monthly Savings' by just $200 in the simulator above to see how it moves your health score!")

    # ============ DATA EXPLORER ============
    elif page == "🔍 Data Explorer":
        st.header("Financial Dataset Explorer")
        st.info("Directly interact with the underlying data and view detailed distributions.")
        
        data_choice = st.selectbox("Select Dataset", ["User Profiles", "Monthly Transactions"])
        
        if data_choice == "User Profiles":
            with st.container(border=True):
                st.markdown("<p class='card-title'>🔍 User Profiles Filter</p>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_income = st.slider("Min Income ($)", 0, 20000, 0)
                with col2:
                    max_income = st.slider("Max Income ($)", 0, 20000, 20000)
                with col3:
                    risk_filter = st.multiselect("Risk Category", 
                                                ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                                default=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                filtered = users_df[
                    (users_df['monthly_income'] >= min_income) &
                    (users_df['monthly_income'] <= max_income) &
                    (users_df['risk_label'].isin(risk_filter))
                ]
                st.dataframe(filtered.head(100), use_container_width=True)
                st.write(f"Showing **{len(filtered)}** of **{len(users_df)}** users.")
        else:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📊 Monthly Transactions Filter</p>", unsafe_allow_html=True)
                month_filter = st.selectbox("Select Month", list(range(1, 13)), format_func=lambda x: f"Month {x}")
                filtered = monthly_df[monthly_df['month'] == month_filter]
                st.dataframe(filtered.head(100), use_container_width=True)
                st.write(f"Showing **{len(filtered)}** records for Month {month_filter}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Statistical Summary</p>", unsafe_allow_html=True)
            if data_choice == "User Profiles":
                st.dataframe(users_df.describe(), use_container_width=True)
            else:
                st.dataframe(monthly_df.describe(), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 30px;'>
        <p style='font-weight: 700; font-size: 1rem; color: #0f172a; margin-bottom: 5px;'>💰 AI-Powered Financial Health Advisor v2.0</p>
        <p style='font-size: 0.85rem;'>Advanced ML Pipeline: Pattern Analysis • Risk Prediction • Monte Carlo Projections • Behavioral Insights</p>
        <p style='font-size: 0.75rem; margin-top: 15px;'>Built for Professional Financial Intelligence and Actionable Strategy.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
