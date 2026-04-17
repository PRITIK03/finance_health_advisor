st.set_page_config(
    page_title="Finance Health Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


TOTAL_USERS = 10000 # Define total number of users for synthetic data

@st.cache_data
def load_data(offset: int = 0, limit: int = None):
    """Load or generate data with pagination."""
    users_df, monthly_df = generate_full_dataset(total_users=TOTAL_USERS, offset=offset, limit=limit)
    return users_df, monthly_df


@st.cache_resource
def train_models(users_df, monthly_df):
    """Train ML models."""
    results, pipeline = train_all_models(users_df, monthly_df)
    return results, pipeline


# Header
col1_header, col2_header = st.columns([1, 5])
with col1_header:
    st.image("https://cdn-icons-png.flaticon.com/512/2845/2845812.png", width=80)
with col2_header:
    st.title("Financial Health Advisor")
    st.markdown("<p style='font-size: 1.1rem; color: #64748b; margin-top: -15px;'>AI-Powered Financial Insights & Behavioral Analysis</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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
        ["📊 Dashboard Overview", "📈 Data Visualization Dashboard", "📚 Financial Tips & Education", "🗂️ Document Upload & Storage", "🚨 Stress Test", "👥 Comparison Mode", "👥 User Segmentation", "🎯 Risk Prediction", 
         "📈 Forecasting", "🚨 Anomaly Detection", "💡 Recommendations", "💸 Expense Categorization", "🎯 Goal Planner", "🔮 Predictive Analytics", "🔔 Alerts & Notifications", "🚀 Wealth Projection", "🔥 FIRE Tracker", "💸 Debt Optimizer", "👥 Peer Benchmarking", "🔮 Scenario Simulator", "🔍 Data Explorer"],
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
    # avg_health = users_df['financial_health_score'].mean() # This will be calculated after data load
    # health_status = "Good" if avg_health > 70 else "Fair" if avg_health > 50 else "Poor"
    
    # Micro Bullet for Average Health vs Target
    # st.plotly_chart(visualizer.create_mini_bullet(avg_health, 85, "Avg Health Index", color="#10b981"), use_container_width=True)
    
    # Financial Milestone Badges (New Feature)
    st.markdown("---")
    st.subheader("🏆 Achievements")
    col1_ach, col2_ach = st.columns(2)
    with col1_ach:
        st.markdown("🛡️ **Debt-Free**")
        st.markdown("🔥 **FIRE Starter**")
    with col2_ach:
        st.markdown("🌟 **Elite Credit**")
        st.markdown("🏦 **Wealth Builder**")
    
    # Top 3 Spending Categories Mini Bar
    st.markdown("---")
    # spending_cols = ['Housing', 'Transportation', 'Food', 'Healthcare', 'Entertainment', 'Shopping', 'Education', 'Subscriptions', 'Insurance', 'Miscellaneous']
    # top_spending = monthly_df[spending_cols].mean().sort_values(ascending=False).head(3)
    # st.plotly_chart(visualizer.create_mini_bar(top_spending.index.tolist(), top_spending.values.tolist(), "Top 3 Avg Monthly Spend", color="#f59e0b"), use_container_width=True)
    
    # st.markdown(f"""
    # <div class='sidebar-text'>
    # <b>Health Index:</b> <span class='highlight'>{avg_health:.1f} ({health_status})</span><br>
    # <b>Model Status:</b> <span style='color: #10b981; font-weight: bold;'>Active</span><br>
    # <b>User Base:</b> <span class='highlight'>{len(users_df):,}</span>
    # </div>
    # """, unsafe_allow_html=True)
    
    # Export Data Button
    st.markdown("---")
    st.subheader("Export Center")
    # summary_data = users_df[['user_id', 'age', 'employment_type', 'monthly_income', 'monthly_expenses', 'financial_health_score', 'risk_label']].to_csv(index=False)
    # st.download_button(
    #     label="📥 Export Analysis (CSV)",
    #     data=summary_data,
    #     file_name="financial_health_summary.csv",
    #     mime="text/csv",
    #     use_container_width=True
    # )

# Initialize session state for pagination
if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 1
if 'page_size' not in st.session_state:
    st.session_state['page_size'] = 100 # Default page size

# Pagination controls
st.sidebar.header("Data Pagination")
page_size_options = [50, 100, 200, 500]
st.session_state['page_size'] = st.sidebar.selectbox("Users per page", page_size_options, index=1) # Default to 100

total_pages = (TOTAL_USERS // st.session_state['page_size']) + (1 if TOTAL_USERS % st.session_state['page_size'] > 0 else 0)
st.sidebar.write(f"Page {st.session_state['page_number']} of {total_pages}")

col1_pag, col2_pag = st.sidebar.columns(2)
with col1_pag:
    if st.button("Previous Page", use_container_width=True, disabled=(st.session_state['page_number'] == 1)):
        st.session_state['page_number'] -= 1
        st.experimental_rerun()
with col2_pag:
    if st.button("Next Page", use_container_width=True, disabled=(st.session_state['page_number'] == total_pages)):
        st.session_state['page_number'] += 1
        st.experimental_rerun()

offset = (st.session_state['page_number'] - 1) * st.session_state['page_size']
limit = st.session_state['page_size']

# Load data
with st.spinner(f"Generating synthetic financial data for page {st.session_state['page_number']}..."):
    users_df, monthly_df = load_data(offset=offset, limit=limit)

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

# ============ GOAL PLANNER ============
if page == "🎯 Goal Planner":
    st.header("Personalized Financial Goal Planner")
    st.info("Set, track, and visualize your financial goals. Stay motivated and monitor your progress!")

    if 'goals' not in st.session_state:
        st.session_state['goals'] = []

    # Goal creation form
    if st.session_state.get('show_goal_form', False):
        with st.form("goal_form", clear_on_submit=True):
            name = st.text_input("Goal Name", max_chars=40)
            goal_type = st.selectbox("Goal Type", ["Savings", "Debt Payoff", "Investment", "Other"])
            target = st.number_input("Target Amount ($)", min_value=1, step=100)
            target_date = st.date_input("Target Date")
            submit = st.form_submit_button("Add Goal")
            if submit and name and target > 0:
                st.session_state['goals'].append({
                    'name': name,
                    'type': goal_type,
                    'target': int(target),
                    'target_date': str(target_date),
                    'progress': 0
                })
                st.session_state['show_goal_form'] = False
                st.success(f"Goal '{name}' added!")
    else:
        if st.button("➕ Add New Goal"):
            st.session_state['show_goal_form'] = True

    # List and update goals
    if st.session_state.get('goals'):
        for idx, goal in enumerate(st.session_state['goals']):
            with st.container(border=True):
                st.markdown(f"### 🎯 {goal['name']} <span style='font-size: 1rem; color: #64748b;'>({goal['type']})</span>", unsafe_allow_html=True)
                st.write(f"**Target:** ${goal['target']:,} by {goal['target_date']}")
                progress = st.slider(f"Progress for {goal['name']}", 0, goal['target'], goal.get('progress', 0), key=f'goal_progress_{idx}')
                goal['progress'] = progress
                pct = (progress / goal['target']) * 100 if goal['target'] > 0 else 0
                st.progress(pct / 100, text=f"{pct:.1f}% of target")
                if pct >= 100:
                    st.success("🎉 Goal achieved!")
                if st.button(f"Remove Goal: {goal['name']}", key=f'remove_goal_{idx}'):
                    st.session_state['goals'].pop(idx)
                    st.experimental_rerun()
    else:
        st.info("No goals set yet. Use 'Add New Goal' to get started!")

    st.markdown("---")
    st.caption("Tip: Add goals for savings, debt payoff, investments, or anything you want to track!")

# ============ DASHBOARD OVERVIEW ============
elif page == "📊 Dashboard Overview":
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
