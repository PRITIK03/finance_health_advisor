import streamlit as st

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

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_full_dataset, calculate_financial_health_score
from preprocessing import FinancialDataPreprocessor, prepare_classification_data, calculate_fire_metrics, calculate_debt_paydown, calculate_emergency_fund_stress_test, calculate_subscription_audit, calculate_financial_stress_test, calculate_wealth_projection
from models import train_all_models
from visualizations import FinancialVisualizer, generate_summary_statistics
from recommendations import RecommendationsEngine
from pages.dashboard import render_dashboard


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


# Initialize session state for pagination
if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 1
if 'page_size' not in st.session_state:
    st.session_state['page_size'] = 100 # Default page size

# Pagination controls
page_size_options = [50, 100, 200, 500]
st.session_state['page_size'] = st.sidebar.selectbox("Users per page", page_size_options, index=1) # Default to 100
total_pages = (TOTAL_USERS // st.session_state['page_size']) + (1 if TOTAL_USERS % st.session_state['page_size'] > 0 else 0)
offset = (st.session_state['page_number'] - 1) * st.session_state['page_size']
limit = st.session_state['page_size']

# Load data and initialize objects before sidebar
with st.spinner(f"Generating synthetic financial data for page {st.session_state['page_number']}..."):
    users_df, monthly_df = load_data(offset=offset, limit=limit)
with st.spinner("Training ML models..."):
    results, pipeline = train_models(users_df, monthly_df)
preprocessor = FinancialDataPreprocessor()
users_processed = preprocessor.preprocess_users(users_df)
monthly_processed = preprocessor.preprocess_monthly(monthly_df)
visualizer = FinancialVisualizer(users_df, monthly_df)
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
        ["📊 Dashboard Overview", "📈 Data Visualization Dashboard", "📚 Financial Tips & Education", "🗂️ Document Upload & Storage", "🚨 Stress Test", "👥 Comparison Mode", "👥 User Segmentation", "🎯 Risk Prediction", 
         "📈 Forecasting", "🚨 Anomaly Detection", "💡 Recommendations", "💸 Expense Categorization", "🎯 Goal Planner", "🔮 Predictive Analytics", "🔔 Alerts & Notifications", "🚀 Wealth Projection", "🔥 FIRE Tracker", "💸 Debt Optimizer", "👥 Peer Benchmarking", "🔮 Scenario Simulator", "🔍 Data Explorer"],
        label_visibility="collapsed"
    )
    
    # Custom CSS for enhanced frontend
    st.markdown("""
    <style>
    /* Animated gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        animation: gradientShift 8s ease infinite;
        background-size: 200% 200%;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Card hover effects */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: all 0.3s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
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
    avg_health = users_df['financial_health_score'].mean() # This will be calculated after data load
    health_status = "Good" if avg_health > 70 else "Fair" if avg_health > 50 else "Poor"
    # Micro Bullet for Average Health vs Target
    st.plotly_chart(visualizer.create_mini_bullet(avg_health, 85, "Avg Health Index", color="#10b981"), use_container_width=True)
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

# ============ DASHBOARD OVERVIEW ============
if page == "📊 Dashboard Overview":
    render_dashboard(users_df, monthly_df, recommendations_engine, visualizer)

# ============ STRESS TEST ============
elif page == "🚨 Stress Test":
    st.header("Financial Stress Simulator")
    
    st.info("Simulate various financial shocks to understand their impact on your financial resilience and net worth over time.")
    
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
            scenario_name = st.selectbox(
                "Select Stress Scenario:",
                ["Job Loss (6 months)", "Medical Emergency", "Market Downturn (20% loss)", "Unexpected Large Expense"]
            )
        with col2:
            simulation_months = st.slider(
                "Simulation Duration (Months)",
                6, 60, 24
            )

    stress_results = calculate_financial_stress_test(
        user_row,
        scenario_name=scenario_name,
        simulation_months=simulation_months
    )
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Liquid Assets", f"${stress_results['initial_liquid_assets']:,.0f}")
    with col2:
        st.metric("Final Liquid Assets", f"${stress_results['final_liquid_assets']:,.0f}")
    with col3:
        st.metric("Initial Net Worth", f"${stress_results['initial_net_worth']:,.0f}")
    with col4:
        st.metric("Final Net Worth", f"${stress_results['final_net_worth']:,.0f}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 2.5])
    
    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>🛟 Resilience Summary</p>", unsafe_allow_html=True)
            st.write(f"**Scenario:** {stress_results['scenario_name']}")
            st.info(stress_results['message'])
            st.write(f"**Outcome:** {stress_results['outcome']}")
            st.write(f"**Severity:** {stress_results['severity']}")
            st.write(f"**Months of Resilience:** {stress_results['months_of_resilience']} out of {simulation_months}")
            
            # Gauge for resilience
            resilience_score = (stress_results['months_of_resilience'] / simulation_months) * 100
            st.plotly_chart(
                visualizer.create_gauge_chart(
                    min(100, resilience_score),
                    "Financial Resilience Score",
                    target=100
                ),
                use_container_width=True
            )
    
    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Financial Trajectory Over Time</p>", unsafe_allow_html=True)
            
            fig = go.Figure()
            months = list(range(simulation_months + 1))
            
            fig.add_trace(go.Scatter(
                x=months,
                y=stress_results['liquid_assets_history'],
                mode='lines',
                name='Liquid Assets',
                line=dict(color='#3b82f6')
            ))
            fig.add_trace(go.Scatter(
                x=months,
                y=stress_results['investments_history'],
                mode='lines',
                name='Investments',
                line=dict(color='#10b981')
            ))
            fig.add_trace(go.Scatter(
                x=months,
                y=stress_results['net_worth_history'],
                mode='lines',
                name='Net Worth',
                line=dict(color='#f59e0b', dash='dash')
            ))
            
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Months",
                yaxis_title="Amount ($)",
                hovermode="x unified",
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("<p class='card-title'>💡 Key Takeaways</p>", unsafe_allow_html=True)
        if stress_results['severity'] == "High":
            st.error(f"**Critical:** {stress_results['outcome']} Immediate action is needed to build reserves or reduce liabilities.")
        elif stress_results['severity'] == "Medium":
            st.warning(f"**Warning:** {stress_results['outcome']} Consider strengthening your financial buffer to withstand longer or more severe shocks.")
        else:
            st.success(f"**Good:** {stress_results['outcome']} Your financial position appears resilient to this scenario.")
        
        st.markdown("---")
        st.write("This simulation provides a simplified view. For personalized advice, consult a financial expert.")

# ============ WEALTH PROJECTION ============
elif page == "🚀 Wealth Projection":
    st.header("Wealth Projection Simulator")
    
    st.info("Project your financial future! See how your investments and savings can grow over time with compound interest.")
    
    # Input parameters for wealth projection
    with st.container(border=True):
        st.markdown("<p class='card-title'>⚙️ Projection Parameters</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=0,
                value=10000,
                step=1000
            )
            monthly_contribution = st.number_input(
                "Monthly Contribution ($)",
                min_value=0,
                value=500,
                step=50
            )
        with col2:
            annual_return_rate = st.slider(
                "Annual Return Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=7.0,
                step=0.1,
                format="%.1f"
            ) / 100 # Convert percentage to decimal
            projection_years = st.slider(
                "Projection Years",
                min_value=1,
                max_value=50,
                value=20,
                step=1
            )
    
    # Calculate wealth projection
    projection_results = calculate_wealth_projection(
        initial_capital=float(initial_capital),
        monthly_contribution=float(monthly_contribution),
        annual_return_rate=float(annual_return_rate),
        projection_years=int(projection_years)
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 2.5])
    
    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📈 Projection Summary</p>", unsafe_allow_html=True)
            st.metric("Projected Final Wealth", f"${projection_results['final_wealth']:,.0f}")
            st.write(f"**Initial Capital:** ${initial_capital:,.0f}")
            st.write(f"**Monthly Contribution:** ${monthly_contribution:,.0f}")
            st.write(f"**Annual Return Rate:** {annual_return_rate*100:.1f}%")
            st.write(f"**Projection Period:** {projection_years} years")
            
            # Gauge for wealth growth
            growth_pct = ((projection_results['final_wealth'] - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
            st.plotly_chart(
                visualizer.create_gauge_chart(
                    min(1000, growth_pct), # Cap gauge at 1000% for display
                    "Wealth Growth (%)",
                    target=100 # Target can be adjusted based on typical growth
                ),
                use_container_width=True
            )
    
    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Wealth Growth Over Time</p>", unsafe_allow_html=True)
            
            fig = go.Figure()
            months = list(range(len(projection_results['wealth_history'])))
            
            fig.add_trace(go.Scatter(
                x=[m / 12 for m in months], # Convert months to years for x-axis
                y=projection_results['wealth_history'],
                mode='lines',
                name='Projected Wealth',
                line=dict(color='#10b981', width=3)
            ))
            
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Years",
                yaxis_title="Wealth ($)",
                hovermode="x unified",
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("<p class='card-title'>💡 Insights & Recommendations</p>", unsafe_allow_html=True)
        st.write("This projection demonstrates the power of compound interest. Even small, consistent contributions can lead to significant wealth over long periods.")
        if annual_return_rate < 0.05:
            st.warning("Consider exploring investment options with potentially higher returns, while being mindful of associated risks.")
        if monthly_contribution == 0:
            st.info("Regular contributions significantly boost wealth growth. Even a small monthly amount can make a big difference.")
        st.markdown("---")
        st.write("Remember that these are projections based on assumptions. Actual returns may vary. Consult a financial advisor for personalized investment strategies.")

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
    
    monthly_data = monthly_df[monthly_df['user_id'] == selected_user_id]
    avg_savings = monthly_data['savings'].mean() if not monthly_data.empty else 0
    
    with st.container(border=True):
        st.markdown("<p class='card-title'>Goal Parameters</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            target_amount = st.number_input("Target Amount ($)", min_value=1000, value=int(user_row['monthly_income'] * 6), step=1000)
        with col2:
            monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=int(avg_savings), step=100)
        with col3:
            expected_return = st.slider("Expected Annual Return (%)", 0.0, 15.0, 7.0, 0.5) / 100
    
    # Calculate projection
    monthly_income_val = float(user_row['monthly_income']) if user_row['monthly_income'] > 0 else 1
    current_savings = float(user_row['monthly_savings'] * 12) if user_row['monthly_savings'] > 0 else float(monthly_income_val * 0.1)
    months_to_goal = 0
    temp_amount = current_savings
    
    if monthly_contribution > 0 or current_savings > 0:
        monthly_rate = (1 + expected_return) ** (1/12) - 1
        while temp_amount < target_amount and months_to_goal < 600:
            temp_amount = (temp_amount + monthly_contribution) * (1 + monthly_rate)
            months_to_goal += 1
    
    years_to_goal = months_to_goal / 12
    goal_date = pd.Timestamp.now() + pd.DateOffset(months=months_to_goal)
    progress_pct = min(100, (current_savings / target_amount) * 100) if target_amount > 0 else 0
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    
    with col1:
        with st.container(border=True):
            st.markdown("<p class='card-title'>Progress Overview</p>", unsafe_allow_html=True)
            st.metric("Current Savings", f"${current_savings:,.0f}")
            st.metric("Target Amount", f"${target_amount:,.0f}")
            st.metric("Progress", f"{progress_pct:.1f}%")
            st.progress(progress_pct / 100, text=f"{progress_pct:.1f}%")
    
    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>Timeline Projection</p>", unsafe_allow_html=True)
            if months_to_goal >= 600:
                st.error("Goal may be unreachable with current contribution rate")
            else:
                st.metric("Months to Goal", f"{months_to_goal}")
                st.metric("Years to Goal", f"{years_to_goal:.1f}")
                st.metric("Estimated Achievement", goal_date.strftime("%B %Y"))
                st.success(f"At ${monthly_contribution:,.0f}/month, you'll reach your goal by {goal_date.strftime('%B %Y')}!")
    
    with col3:
        with st.container(border=True):
            st.markdown("<p class='card-title'>Optimization Tips</p>", unsafe_allow_html=True)
            if monthly_contribution < target_amount * 0.02:
                st.warning("Consider increasing monthly contributions")
            if expected_return < 0.05:
                st.info("Higher returns could accelerate your timeline")
            if progress_pct < 10:
                st.info("Starting is the hardest part - keep going!")
            st.caption("Adjust parameters to see real-time updates")

# ============ FIRE TRACKER ============
elif page == "🔥 FIRE Tracker":
    st.header("Financial Independence, Retire Early (FIRE) Tracker")
    
    st.info("Calculate your path to financial independence and see when you can retire early based on your current savings rate and investments.")
    
    selected_user_id = st.selectbox(
        "Select User Profile for FIRE Analysis",
        users_df['user_id'].unique(),
        format_func=lambda x: f"User {x}"
    )
    user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]
    
    # FIRE Settings
    with st.container(border=True):
        st.markdown("<p class='card-title'>FIRE Parameters</p>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            safe_withdrawal = st.slider("Safe Withdrawal Rate (%)", 2.0, 6.0, 4.0, 0.1) / 100
        with col2:
            expected_return = st.slider("Expected Return (%)", 3.0, 12.0, 7.0, 0.5) / 100
        with col3:
            inflation_rate = st.slider("Inflation Rate (%)", 0.0, 5.0, 3.0, 0.1) / 100
        with col4:
            current_investments = st.number_input(
                "Current Investments ($)", 
                min_value=0, 
                value=int(user_row.get('monthly_investments', 0) * 24),
                step=1000
            )
    
    # Calculate FIRE metrics
    annual_expenses = float(user_row['monthly_expenses']) * 12
    monthly_contribution = float(user_row['monthly_savings']) + float(user_row['monthly_investments'])
    annual_contribution = monthly_contribution * 12
    real_return = expected_return - inflation_rate
    
    # Recalculate years to FIRE
    years_to_fire = 0
    temp_wealth = current_investments
    fire_number = annual_expenses / safe_withdrawal if safe_withdrawal > 0 else float('inf')
    
    if annual_contribution > 0 and real_return > 0:
        while temp_wealth < fire_number and years_to_fire < 100:
            temp_wealth = (temp_wealth + annual_contribution) * (1 + real_return)
            years_to_fire += 1
    elif annual_contribution > 0:
        years_to_fire = (fire_number - current_investments) / annual_contribution if annual_contribution > 0 else float('inf')
    
    lean_fire = annual_expenses * 0.7 / safe_withdrawal if safe_withdrawal > 0 else 0
    fat_fire = annual_expenses * 1.5 / safe_withdrawal if safe_withdrawal > 0 else 0
    progress_pct = min(100, (current_investments / fire_number) * 100) if fire_number > 0 else 0
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FIRE Number", f"${fire_number:,.0f}", help="Amount needed to retire with current expenses")
    with col2:
        st.metric("Years to FIRE", f"{years_to_fire}", help="Estimated years until financial independence")
    with col3:
        st.metric("Current Progress", f"{progress_pct:.1f}%", help="Percentage of FIRE goal achieved")
    with col4:
        st.metric("Monthly Investment", f"${monthly_contribution:,.0f}", help="Current monthly savings + investments")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["FIRE Analysis", "Growth Projection", "FIRE Scenarios"])
    
    with tab1:
        col1, col2 = st.columns([1.5, 2.5])
        
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Your FIRE Targets</p>", unsafe_allow_html=True)
                
                lean_progress = min(100, (current_investments / lean_fire) * 100) if lean_fire > 0 else 0
                st.markdown(f"**Lean FIRE:** ${lean_fire:,.0f} (70% of expenses)")
                st.progress(lean_progress / 100, text=f"{lean_progress:.1f}%")
                
                st.markdown(f"**Regular FIRE:** ${fire_number:,.0f} (100% of expenses)")
                st.progress(progress_pct / 100, text=f"{progress_pct:.1f}%")
                
                fat_progress = min(100, (current_investments / fat_fire) * 100) if fat_fire > 0 else 0
                st.markdown(f"**Fat FIRE:** ${fat_fire:,.0f} (150% of expenses)")
                st.progress(fat_progress / 100, text=f"{fat_progress:.1f}%")
                
                st.markdown("---")
                st.markdown(f"**Annual Expenses:** ${annual_expenses:,.0f}")
                st.markdown(f"**Safe Withdrawal:** {safe_withdrawal*100:.1f}% = ${annual_expenses:,.0f}/year")
            
            with st.container(border=True):
                st.markdown("<p class='card-title'>FIRE Readiness Score</p>", unsafe_allow_html=True)
                st.plotly_chart(
                    visualizer.create_gauge_chart(min(100, progress_pct * 2), "FIRE Readiness", target=100),
                    use_container_width=True
                )
        
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>FIRE Milestone Timeline</p>", unsafe_allow_html=True)
                
                milestones = []
                milestone_names = ["25%", "50%", "75%", "90%", "100% (FIRE)"]
                milestone_values = [fire_number * p for p in [0.25, 0.5, 0.75, 0.9, 1.0]]
                
                for name, target in zip(milestone_names, milestone_values):
                    if current_investments >= target:
                        milestones.append({"Milestone": name, "Status": "Achieved", "Target": f"${target:,.0f}"})
                    else:
                        remaining = target - current_investments
                        if annual_contribution > 0 and real_return > 0:
                            years_remaining = 0
                            temp = current_investments
                            while temp < target and years_remaining < 100:
                                temp = (temp + annual_contribution) * (1 + real_return)
                                years_remaining += 1
                            milestones.append({"Milestone": name, "Status": f"{years_remaining} years", "Target": f"${target:,.0f}"})
                        else:
                            milestones.append({"Milestone": name, "Status": "Not achievable", "Target": f"${target:,.0f}"})
                
                st.dataframe(pd.DataFrame(milestones), use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("<p class='card-title'>Savings Rate Impact</p>", unsafe_allow_html=True)
                
                current_savings_rate = (user_row['monthly_savings'] / user_row['monthly_income'] * 100) if user_row['monthly_income'] > 0 else 0
                
                if current_savings_rate < 20:
                    st.warning(f"Your savings rate is {current_savings_rate:.1f}%. Increasing to 50%+ could accelerate FIRE.")
                elif current_savings_rate < 50:
                    st.info(f"Good progress! At {current_savings_rate:.1f}% savings rate, you're on the path to FIRE.")
                else:
                    st.success(f"Excellent! Your {current_savings_rate:.1f}% savings rate puts you on the fast track to FIRE!")
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Net Worth Projection to FIRE</p>", unsafe_allow_html=True)
                
                years = list(range(min(years_to_fire + 6, 51)))
                wealth_values = []
                temp = current_investments
                for _ in years:
                    wealth_values.append(temp)
                    temp = (temp + annual_contribution) * (1 + real_return)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years, y=wealth_values,
                    mode='lines+markers',
                    name='Projected Net Worth',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=6)
                ))
                
                fig.add_hline(y=fire_number, line_dash="dash", line_color="#ef4444", 
                             annotation_text="FIRE Number", annotation_position="top right")
                
                fig.add_hline(y=lean_fire, line_dash="dot", line_color="#f59e0b",
                             annotation_text="Lean FIRE", annotation_position="bottom right")
                
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Years from Now",
                    yaxis_title="Net Worth ($)",
                    hovermode="x unified",
                    margin=dict(t=30, b=30, l=30, r=30),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Contribution Analysis</p>", unsafe_allow_html=True)
                
                st.metric("Current Investments", f"${current_investments:,.0f}")
                st.metric("Annual Contribution", f"${annual_contribution:,.0f}")
                st.metric("Real Return Rate", f"{real_return*100:.1f}%")
                
                st.markdown("---")
                st.markdown("**Quick Adjustments:**")
                
                extra_100 = 100 * 12
                temp_with_extra = current_investments
                years_with_extra = 0
                while temp_with_extra < fire_number and years_with_extra < 100:
                    temp_with_extra = (temp_with_extra + annual_contribution + extra_100) * (1 + real_return)
                    years_with_extra += 1
                
                time_saved = years_to_fire - years_with_extra
                if time_saved > 0:
                    st.success(f"+${extra_100/12:.0f}/month saves {time_saved} years!")
    
    with tab3:
        with st.container(border=True):
            st.markdown("<p class='card-title'>Compare FIRE Scenarios</p>", unsafe_allow_html=True)
            
            scenarios = []
            for withdrawal in [0.03, 0.04, 0.05]:
                for return_rate in [0.05, 0.07, 0.09]:
                    real = return_rate - inflation_rate
                    fire_num = annual_expenses / withdrawal if withdrawal > 0 else float('inf')
                    
                    years = 0
                    temp = current_investments
                    annual = annual_contribution
                    while temp < fire_num and years < 100:
                        temp = (temp + annual) * (1 + real)
                        years += 1
                    
                    scenarios.append({
                        "Withdrawal Rate": f"{withdrawal*100:.0f}%",
                        "Return Rate": f"{return_rate*100:.0f}%",
                        "FIRE Number": f"${fire_num:,.0f}",
                        "Years to FIRE": years if years < 100 else "100+",
                        "Monthly Income": f"${fire_num * withdrawal / 12:,.0f}"
                    })
            
            st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)
            
            st.info("Key Insight: Lower withdrawal rates require more savings but provide more safety. Higher returns accelerate your timeline but come with more volatility.")

# ============ DEBT OPTIMIZER ============
elif page == "💸 Debt Optimizer":
    st.header("Smart Debt Optimizer")
    
    st.info("Compare debt payoff strategies and find the optimal approach to eliminate your debt faster while saving on interest payments.")
    
    selected_user_id = st.selectbox(
        "Select User Profile for Debt Analysis",
        users_df['user_id'].unique(),
        format_func=lambda x: f"User {x}"
    )
    user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]
    
    default_debts = [
        {"name": "Credit Card", "balance": float(user_row['total_debt']) * 0.4 if user_row['total_debt'] > 0 else 5000, 
         "interest_rate": 19.99, "min_payment": 150},
        {"name": "Car Loan", "balance": float(user_row['total_debt']) * 0.35 if user_row['total_debt'] > 0 else 8000, 
         "interest_rate": 5.99, "min_payment": 300},
        {"name": "Student Loan", "balance": float(user_row['total_debt']) * 0.25 if user_row['total_debt'] > 0 else 6000, 
         "interest_rate": 4.5, "min_payment": 200},
    ]
    
    with st.container(border=True):
        st.markdown("<p class='card-title'>Debt Configuration</p>", unsafe_allow_html=True)
        
        debts = []
        for i, default in enumerate(default_debts):
            with st.expander(f"Debt {i+1}: {default['name']}", expanded=i==0):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    name = st.text_input(f"Name {i}", value=default['name'], key=f"debt_name_{i}")
                with col2:
                    balance = st.number_input(f"Balance ${i}", min_value=0.0, value=float(default['balance']), step=100.0, key=f"debt_bal_{i}")
                with col3:
                    rate = st.number_input(f"APR % {i}", min_value=0.0, max_value=50.0, value=float(default['interest_rate']), step=0.1, key=f"debt_rate_{i}")
                with col4:
                    min_pay = st.number_input(f"Min Pay ${i}", min_value=0.0, value=float(default['min_payment']), step=10.0, key=f"debt_min_{i}")
                
                if balance > 0:
                    debts.append({"name": name, "balance": balance, "interest_rate": rate, "min_payment": min_pay})
        
        col1, col2 = st.columns(2)
        with col1:
            extra_payment = st.number_input("Extra Monthly Payment ($)", min_value=0.0, value=float(user_row['monthly_savings'] * 0.5), step=50.0,
                                           help="Additional amount to put toward debt payoff each month")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            optimize_button = st.button("Calculate Optimal Strategy", use_container_width=True, type="primary")
    
    if debts and optimize_button:
        results = calculate_debt_paydown(debts, extra_payment)
        
        avalanche = results['avalanche']
        snowball = results['snowball']
        
        total_debt = sum(d['balance'] for d in debts)
        min_monthly = sum(d['min_payment'] for d in debts)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Debt", f"${total_debt:,.0f}")
        with col2:
            st.metric("Min. Monthly Payment", f"${min_monthly:,.0f}")
        with col3:
            st.metric("Extra Payment", f"${extra_payment:,.0f}")
        with col4:
            st.metric("Total Monthly", f"${min_monthly + extra_payment:,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("Strategy Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Avalanche Method (Highest Interest First)</p>", unsafe_allow_html=True)
                
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.metric("Payoff Time", f"{avalanche['months']} months")
                    st.metric("Total Interest", f"${avalanche['total_interest']:,.0f}")
                with col_a2:
                    st.metric("Years to Debt-Free", f"{avalanche['months']/12:.1f}")
                    if avalanche['months'] > 0:
                        avg_monthly = (total_debt + avalanche['total_interest']) / avalanche['months']
                        st.metric("Avg Monthly Cost", f"${avg_monthly:,.0f}")
                
                st.markdown("---")
                st.markdown("**Priority Order:** Pay minimums on all, put extra toward highest interest debt first.")
                st.success("Best for: Saving the most money on interest")
        
        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Snowball Method (Smallest Balance First)</p>", unsafe_allow_html=True)
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Payoff Time", f"{snowball['months']} months")
                    st.metric("Total Interest", f"${snowball['total_interest']:,.0f}")
                with col_s2:
                    st.metric("Years to Debt-Free", f"{snowball['months']/12:.1f}")
                    if snowball['months'] > 0:
                        avg_monthly = (total_debt + snowball['total_interest']) / snowball['months']
                        st.metric("Avg Monthly Cost", f"${avg_monthly:,.0f}")
                
                st.markdown("---")
                st.markdown("**Priority Order:** Pay minimums on all, put extra toward smallest balance first.")
                st.info("Best for: Quick wins and motivation")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        interest_saved = snowball['total_interest'] - avalanche['total_interest']
        months_diff = snowball['months'] - avalanche['months']
        
        with st.container(border=True):
            st.markdown("<p class='card-title'>AI Recommendation</p>", unsafe_allow_html=True)
            
            if interest_saved > 100:
                st.success(f"Recommended: Avalanche Method - You'll save ${interest_saved:,.0f} in interest compared to Snowball!")
            elif months_diff > 3:
                st.info(f"Either method works, but Avalanche saves ${interest_saved:,.0f} and pays off {months_diff} months faster.")
            else:
                st.success("Both strategies are similar for your debt profile. Choose based on what motivates you more!")
            
            st.markdown("---")
            
            debt_free_date = datetime.now() + timedelta(days=avalanche['months'] * 30)
            st.markdown(f"Estimated Debt-Free Date: {debt_free_date.strftime('%B %Y')} (using Avalanche)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Balance Over Time", "Debt Breakdown"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>Avalanche Payoff Timeline</p>", unsafe_allow_html=True)
                    
                    if avalanche['history']:
                        months = list(range(len(avalanche['history'])))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=months, y=avalanche['history'],
                            mode='lines', fill='tozeroy',
                            name='Remaining Balance',
                            line=dict(color='#3b82f6', width=2),
                            fillcolor='rgba(59, 130, 246, 0.2)'
                        ))
                        fig.update_layout(
                            template="plotly_white",
                            xaxis_title="Months",
                            yaxis_title="Remaining Balance ($)",
                            margin=dict(t=20, b=20, l=20, r=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>Snowball Payoff Timeline</p>", unsafe_allow_html=True)
                    
                    if snowball['history']:
                        months = list(range(len(snowball['history'])))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=months, y=snowball['history'],
                            mode='lines', fill='tozeroy',
                            name='Remaining Balance',
                            line=dict(color='#10b981', width=2),
                            fillcolor='rgba(16, 185, 129, 0.2)'
                        ))
                        fig.update_layout(
                            template="plotly_white",
                            xaxis_title="Months",
                            yaxis_title="Remaining Balance ($)",
                            margin=dict(t=20, b=20, l=20, r=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Your Debt Portfolio</p>", unsafe_allow_html=True)
                
                debt_df = pd.DataFrame(debts)
                debt_df['Annual Interest'] = debt_df['balance'] * (debt_df['interest_rate'] / 100)
                debt_df['Cost per Month'] = debt_df['Annual Interest'] / 12
                
                st.dataframe(
                    debt_df[['name', 'balance', 'interest_rate', 'min_payment', 'Cost per Month']]
                    .style.format({
                        'balance': '${:,.0f}',
                        'interest_rate': '{:.2f}%',
                        'min_payment': '${:,.0f}',
                        'Cost per Month': '${:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                total_annual_interest = debt_df['Annual Interest'].sum()
                st.markdown(f"**Total Annual Interest Cost:** ${total_annual_interest:,.0f} (${total_annual_interest/12:,.0f}/month)")
    
    else:
        st.info("Configure your debts above and click 'Calculate Optimal Strategy' to see your personalized payoff plan.")

# ============ DATA VISUALIZATION DASHBOARD ============
elif page == "📈 Data Visualization Dashboard":
    st.header("📈 Comprehensive Data Visualization Dashboard")
    st.info("Interactive visualizations for deep financial data exploration.")
    
    tabs = st.tabs(["Distributions", "Correlations", "Trends", "Employment", "Anomalies"])
    
    with tabs[0]:
        st.plotly_chart(visualizer.create_distribution_plots()['distribution_plots'], use_container_width=True)
    
    with tabs[1]:
        st.plotly_chart(visualizer.create_correlation_heatmap(), use_container_width=True)
    
    with tabs[2]:
        st.plotly_chart(visualizer.create_time_series_plot(), use_container_width=True)
    
    with tabs[3]:
        st.plotly_chart(visualizer.create_employment_analysis(), use_container_width=True)
    
    with tabs[4]:
        st.plotly_chart(visualizer.create_anomaly_analysis_plot(), use_container_width=True)

# ============ FINANCIAL TIPS & EDUCATION ============
elif page == "📚 Financial Tips & Education":
    st.header("📚 Financial Literacy & Education Center")
    st.info("Evidence-based financial education modules powered by behavioral insights.")
    
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

# ============ DOCUMENT UPLOAD & STORAGE ============
elif page == "🗂️ Document Upload & Storage":
    st.header("🗂️ Secure Document Vault (Demo)")
    st.warning("⚠️ This is a demo. In production, use encrypted storage and proper auth.")
    
    uploaded_files = st.file_uploader(
        "Upload financial documents (PDF, CSV, images)",
        type=["pdf", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s) (demo only — not persisted).")
        for f in uploaded_files:
            st.write(f"• {f.name} ({f.size} bytes)")

# ============ EXPENSE CATEGORIZATION ============
elif page == "💸 Expense Categorization":
    st.header("💸 AI Expense Categorization")
    st.info("Upload transactions or view auto-categorized spending from the synthetic dataset.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Category Breakdown (Sample)")
        spending_cols = ['Housing', 'Transportation', 'Food', 'Healthcare', 'Entertainment', 'Shopping', 'Education', 'Subscriptions', 'Insurance', 'Miscellaneous']
        avg_spend = monthly_df[spending_cols].mean().sort_values(ascending=False)
        st.dataframe(avg_spend.reset_index().rename(columns={'index': 'Category', 0: 'Avg Monthly $'}))
    
    with col2:
        st.plotly_chart(visualizer.create_spending_breakdown_chart(), use_container_width=True)

# ============ PREDICTIVE ANALYTICS ============
elif page == "🔮 Predictive Analytics":
    st.header("🔮 Predictive Analytics Suite")
    st.info("Advanced forecasting and scenario modeling (Monte Carlo + ML).")
    
    st.subheader("Savings Forecast (Gradient Boosting)")
    st.metric("R² Score", f"{results['forecasting']['test_metrics']['r2']:.3f}")
    st.metric("RMSE", f"${results['forecasting']['test_metrics']['rmse']:.0f}")
    st.plotly_chart(visualizer.create_time_series_plot(), use_container_width=True)

# ============ ALERTS & NOTIFICATIONS ============
elif page == "🔔 Alerts & Notifications":
    st.header("🔔 Smart Alerts & Notifications")
    
    st.subheader("Critical Alerts (Demo)")
    alert_users = users_df[
        (users_df['monthly_savings'] / users_df['monthly_income'].replace(0, np.nan) < 0.05) |
        (users_df['monthly_expenses'] / users_df['monthly_income'].replace(0, np.nan) > 0.85)
    ]
    
    if not alert_users.empty:
        for _, row in alert_users.head(5).iterrows():
            st.error(f"User {row['user_id']}: Very low savings rate or dangerously high expense ratio!")
    else:
        st.success("No critical alerts in current page.")

# ============ PEER BENCHMARKING ============
elif page == "👥 Peer Benchmarking":
    st.header("👥 Peer Benchmarking")
    st.info("Compare yourself to similar financial profiles using KNN.")
    
    user_id = st.selectbox("Select User", users_df['user_id'].unique())
    user_row = users_df[users_df['user_id'] == user_id].iloc[0]
    
    st.plotly_chart(visualizer.create_gauge_chart(user_row['financial_health_score'], f"User {user_id} Health"), use_container_width=True)
    st.caption("Peer comparison radar coming from KNN model (see models.py).")

# ============ SCENARIO SIMULATOR ============
elif page == "🔮 Scenario Simulator":
    st.header("🔮 What-If Scenario Simulator")
    st.info("Run custom Monte Carlo simulations on your financial future.")
    
    st.write("**Coming soon:** Full Monte Carlo wealth simulator with adjustable volatility, sequence-of-returns risk, and custom life events.")

# ============ DATA EXPLORER ============
elif page == "🔍 Data Explorer":
    st.header("🔍 Raw Data Explorer")
    
    tab1, tab2 = st.tabs(["Users", "Monthly Transactions"])
    
    with tab1:
        st.dataframe(users_df, use_container_width=True, height=500)
        st.download_button("Download Users CSV", users_df.to_csv(index=False), "users.csv", "text/csv")
    
    with tab2:
        st.dataframe(monthly_df, use_container_width=True, height=500)
        st.download_button("Download Monthly CSV", monthly_df.to_csv(index=False), "monthly.csv", "text/csv")

# ============ SAFETY NET ============
else:
    st.warning("🚧 This section is under active development.")
    st.info("Please select a different module from the sidebar.")
