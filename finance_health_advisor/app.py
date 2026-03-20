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
from preprocessing import FinancialDataPreprocessor, prepare_classification_data
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b !important;
    }
    
    .stApp {
        background-color: #f8f9fa !important;
    }

    /* Fix Metric visibility */
    [data-testid="stMetric"] {
        background-color: white !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        border: 1px solid #e2e8f0 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        margin-bottom: 5px !important;
    }
    
    /* Container styling (using st.container()) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 24px !important;
        padding: 10px !important;
    }

    /* Legacy Card for backward compatibility or direct HTML */
    .custom-card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 25px;
        color: #1e293b;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 20px;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 12px;
    }
    
    .sidebar-text {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.6;
    }
    
    .highlight {
        color: #2563eb;
        font-weight: 700;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        border-bottom: 2px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: transparent;
        color: #64748b;
        font-weight: 600;
        font-size: 1rem;
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }

    /* Hide redundant elements if any */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Content Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 95% !important;
    }

    /* Ensure text inside markdown and headers is dark */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    
    /* Fix Sidebar text visibility */
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #475569 !important;
    }
    
    /* Fix for Streamlit's info/warning boxes to keep their intended contrast */
    [data-testid="stNotification"] p {
        color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2845/2845812.png", width=100)
    with col2:
        st.title("Financial Health Advisor")
        st.markdown("<p style='font-size: 1.2rem; color: #64748b;'>AI-Powered Insights for Your Financial Future</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2845/2845812.png", width=60)
    st.sidebar.title("Navigation")
    
    # Sidebar Info
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.markdown("""
    <div class='sidebar-text'>
    This advisor uses <span class='highlight'>Advanced Machine Learning</span> to analyze financial patterns, 
    predict risks, and provide actionable insights.
    <br><br>
    <b>Key Features:</b>
    <ul>
        <li>User Segmentation</li>
        <li>Risk Prediction</li>
        <li>Savings Forecasting</li>
        <li>Anomaly Detection</li>
        <li>Personalized Recommendations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Navigation
    page = st.sidebar.radio(
        "Select Section",
        ["📊 Dashboard Overview", "👥 User Segmentation", "🎯 Risk Prediction", 
         "📈 Forecasting", "🚨 Anomaly Detection", "💡 Recommendations", "👥 Peer Benchmarking", "🔮 Scenario Simulator", "🔍 Data Explorer"]
    )
    
    # Sidebar Info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Insights")
    avg_health = users_df['financial_health_score'].mean()
    health_status = "Good" if avg_health > 70 else "Fair" if avg_health > 50 else "Poor"
    
    st.sidebar.markdown(f"""
    <div class='sidebar-text'>
    <b>System Status:</b> <span style='color: #10b981;'>Live</span><br>
    <b>Avg Health:</b> <span class='highlight'>{avg_health:.1f} ({health_status})</span><br>
    <b>Risk Focus:</b> <span class='highlight'>{users_df['risk_label'].mode()[0]}</span><br>
    <b>Model Accuracy:</b> <span class='highlight'>{results['classification']['test_metrics']['accuracy']:.1%}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Export Data Button
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Analysis")
    summary_data = users_df[['user_id', 'age', 'employment_type', 'monthly_income', 'monthly_expenses', 'financial_health_score', 'risk_label']].to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download User Summary (CSV)",
        data=summary_data,
        file_name="financial_health_summary.csv",
        mime="text/csv",
    )
    
    # ============ DASHBOARD OVERVIEW ============
    if page == "📊 Dashboard Overview":
        st.header("Financial Overview")
        
        # Summary Statistics
        stats = generate_summary_statistics(users_df, monthly_df)
        
        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{stats['total_users']:,}", delta=None)
        with col2:
            st.metric("Avg Monthly Income", f"${stats['avg_income']:,.0f}", delta=None)
        with col3:
            st.metric("Avg Savings", f"${stats['avg_savings']:,.0f}", delta=None)
        with col4:
            st.metric("Avg Health Score", f"{stats['avg_health_score']:.1f}/100", delta=None)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Key Metrics", "🔍 Relationship Analysis", "📋 Summary Data"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.plotly_chart(visualizer.create_risk_analysis_plot(), use_container_width=True)
            
            with col2:
                with st.container():
                    st.plotly_chart(visualizer.create_spending_breakdown_chart(), use_container_width=True)
            
            with st.container():
                st.plotly_chart(visualizer.create_time_series_plot(), use_container_width=True)
                
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.plotly_chart(visualizer.create_income_expense_scatter(), use_container_width=True)
            
            with col2:
                with st.container():
                    st.plotly_chart(visualizer.create_employment_analysis(), use_container_width=True)
                    
            with st.container():
                st.plotly_chart(visualizer.create_correlation_heatmap(), use_container_width=True)
                
        with tab3:
            with st.container():
                st.subheader("Recent Financial Snapshot")
                st.dataframe(users_df[['user_id', 'age', 'employment_type', 'monthly_income', 'monthly_expenses', 'financial_health_score']].head(20), use_container_width=True)
    
    # ============ USER SEGMENTATION ============
    elif page == "👥 User Segmentation":
        st.header("User Segmentation Analysis")
        
        st.info("K-Means clustering identifies 5 distinct financial profiles based on behavior and wealth metrics.")
        
        tab1, tab2 = st.tabs(["📊 Cluster Overview", "📋 Detailed Statistics"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.container():
                    st.plotly_chart(visualizer.create_cluster_analysis_plot(), use_container_width=True)
                
            with col2:
                with st.container():
                    st.markdown("<p class='card-title'>Cluster Distribution</p>", unsafe_allow_html=True)
                    cluster_counts = users_df['cluster'].value_counts().sort_index()
                    for i, count in cluster_counts.items():
                        st.write(f"**Cluster {i}:** {count} users")
        
        with tab2:
            with st.container():
                st.subheader("Cluster Profiles")
                cluster_stats = users_df.groupby('cluster').agg({
                    'monthly_income': 'mean',
                    'monthly_expenses': 'mean',
                    'monthly_savings': 'mean',
                    'credit_score': 'mean',
                    'financial_health_score': 'mean'
                }).round(2)
                st.dataframe(cluster_stats, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Profile Descriptions")
            cols = st.columns(3)
            for i, cluster_id in enumerate(sorted(users_df['cluster'].unique())):
                with cols[i % 3]:
                    cluster_data = users_df[users_df['cluster'] == cluster_id]
                    with st.container():
                        st.markdown(f"<p class='card-title'>Profile {cluster_id}</p>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class='sidebar-text'>
                        Avg Health: <span class='highlight'>{cluster_data['financial_health_score'].mean():.1f}</span><br>
                        Avg Income: ${cluster_data['monthly_income'].mean():,.0f}<br>
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
                with st.container():
                    st.markdown("<p class='card-title'>Risk Distribution</p>", unsafe_allow_html=True)
                    risk_counts = users_df['risk_label'].value_counts()
                    colors = {'Very Low': '#2ecc71', 'Low': '#27ae60', 'Medium': '#f1c40f', 
                             'High': '#e67e22', 'Very High': '#e74c3c'}
                    
                    fig = px.pie(
                        values=risk_counts.values, 
                        names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map=colors,
                        hole=0.5
                    )
                    fig.update_layout(
                        template="plotly_white", 
                        margin=dict(t=20, b=20, l=20, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                with st.container():
                    st.markdown("<p class='card-title'>Risk Insights</p>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='sidebar-text'>
                    The model classifies users into 5 risk levels. Most users fall into the <span class='highlight'>Medium</span> to <span class='highlight'>Low</span> risk categories.
                    <br><br>
                    <b>Key drivers for risk include:</b>
                    <ul>
                        <li>Credit Score</li>
                        <li>Monthly Savings</li>
                        <li>Total Debt</li>
                        <li>Income Stability</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
        with tab2:
            with st.container():
                st.markdown("<p class='card-title'>Feature Importance</p>", unsafe_allow_html=True)
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
                    height=500
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
            with st.container():
                st.markdown("<p class='card-title'>Monthly Savings Trends</p>", unsafe_allow_html=True)
                
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
                    line=dict(color='#3b82f6', width=3, shape='spline')
                ))
                
                # Add trend line
                z = np.polyfit(monthly_avg['month'], monthly_avg['savings'], 2)
                p = np.poly1d(z)
                x_trend = np.arange(1, 13)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Forecast Trend',
                    line=dict(color='#ef4444', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title='Month',
                    yaxis_title='Savings ($)',
                    xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                    margin=dict(t=20, b=20, l=20, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            with st.container():
                st.markdown("<p class='card-title'>Savings Patterns by Risk Category</p>", unsafe_allow_html=True)
                
                savings_by_risk = monthly_df.groupby('risk_label')['savings'].mean().sort_values()
                
                fig = px.bar(
                    x=savings_by_risk.index,
                    y=savings_by_risk.values,
                    labels={'x': 'Risk Category', 'y': 'Average Savings ($)'},
                    color=savings_by_risk.values,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(template="plotly_white", margin=dict(t=20, b=20, l=20, r=20))
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
            with st.container():
                st.plotly_chart(visualizer.create_anomaly_analysis_plot(), use_container_width=True)
            
        with tab2:
            with st.container():
                st.markdown("<p class='card-title'>Sample of Flagged Transactions</p>", unsafe_allow_html=True)
                
                anomalies = monthly_df[monthly_df['ml_anomaly'] == 1]
                st.dataframe(
                    anomalies[['user_id', 'month', 'income', 'expenses', 'savings', 'Entertainment', 'Shopping']].head(20),
                    use_container_width=True
                )
    
    # ============ RECOMMENDATIONS ============
    elif page == "💡 Recommendations":
        st.header("Personalized Financial Recommendations")
        
        st.info("AI-powered recommendations tailored to your financial profile based on ML analysis.")
        
        # Recommendation type selector
        rec_type = st.radio(
            "Select Recommendation Type",
            ["👤 Individual User", "👥 Cohort Analysis"]
        )
        
        if rec_type == "👤 Individual User":
            # User selector
            user_id = st.selectbox(
                "Select a User",
                users_df['user_id'].unique(),
                format_func=lambda x: f"User {x}"
            )
            
            # Get user data
            user_row = users_df[users_df['user_id'] == user_id].iloc[0]
            
            # Display user summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Income", f"${user_row['monthly_income']:,.0f}")
            with col2:
                st.metric("Expenses", f"${user_row['monthly_expenses']:,.0f}")
            with col3:
                st.metric("Savings", f"${user_row['monthly_savings']:,.0f}")
            with col4:
                st.metric("Health Score", f"{user_row['financial_health_score']:.0f}/100")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Get recommendations
            user_recs = recommendations_engine.generate_user_recommendations(user_id)
            
            # Display recommendations in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["💰 Budget", "💳 Debt", "🏦 Savings", "📈 Investments"])
            
            with tab1:
                with st.container():
                    st.markdown("<p class='card-title'>Budget Recommendations</p>", unsafe_allow_html=True)
                    for rec in user_recs['budget']:
                        status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠'}.get(rec['status'], '⚪')
                        st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                        st.markdown(f"   *💡 {rec['suggestion']}*")
                        st.markdown("")
            
            with tab2:
                with st.container():
                    st.markdown("<p class='card-title'>Debt Management Recommendations</p>", unsafe_allow_html=True)
                    for rec in user_recs['debt']:
                        status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'excellent': '💚', 'info': '🔵'}.get(rec['status'], '⚪')
                        st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                        st.markdown(f"   *💡 {rec['suggestion']}*")
                        st.markdown("")
            
            with tab3:
                with st.container():
                    st.markdown("<p class='card-title'>Savings & Emergency Fund Recommendations</p>", unsafe_allow_html=True)
                    for rec in user_recs['savings']:
                        status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'info': '🔵'}.get(rec['status'], '⚪')
                        st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                        st.markdown(f"   *💡 {rec['suggestion']}*")
                        st.markdown("")
            
            with tab4:
                with st.container():
                    st.markdown("<p class='card-title'>Investment Recommendations</p>", unsafe_allow_html=True)
                    for rec in user_recs['investments']:
                        status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'info': '🔵'}.get(rec['status'], '⚪')
                        st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                        st.markdown(f"   *💡 {rec['suggestion']}*")
                        st.markdown("")
        
        else:
            # Cohort Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_filter = st.selectbox(
                    "Filter by Cluster",
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
                st.markdown(f"*{cohort_recs['summary']}*")
                
                # Display cohort recommendations
                tab1, tab2, tab3, tab4 = st.tabs(["💰 Budget", "💳 Debt", "🏦 Savings", "📈 Investments"])
                
                all_recs = cohort_recs['recommendations']
                
                with tab1:
                    with st.container():
                        st.markdown("<p class='card-title'>Budget Recommendations</p>", unsafe_allow_html=True)
                        for rec in all_recs['budget']:
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")
                
                with tab2:
                    with st.container():
                        st.markdown("<p class='card-title'>Debt Management Recommendations</p>", unsafe_allow_html=True)
                        for rec in all_recs['debt']:
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'excellent': '💚', 'info': '🔵'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")
                
                with tab3:
                    with st.container():
                        st.markdown("<p class='card-title'>Savings & Emergency Fund Recommendations</p>", unsafe_allow_html=True)
                        for rec in all_recs['savings']:
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'info': '🔵'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")
                
                with tab4:
                    with st.container():
                        st.markdown("<p class='card-title'>Investment Recommendations</p>", unsafe_allow_html=True)
                        for rec in all_recs['investments']:
                            status_color = {'good': '🟢', 'warning': '🟡', 'critical': '🔴', 'moderate': '🟠', 'info': '🔵'}.get(rec['status'], '⚪')
                            st.markdown(f"**{status_color} {rec['category']}**: {rec['message']}")
                            st.markdown(f"   *💡 {rec['suggestion']}*")
                            st.markdown("")
            else:
                st.warning("No users found in the selected cohort.")

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
        # Prepare data for KNN (must match clustering features)
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
            with st.container():
                st.markdown("<p class='card-title'>Peer Comparison Radar</p>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_peer_comparison_radar(user_data, peers_avg), use_container_width=True)
        
        with col2:
            with st.container():
                st.markdown("<p class='card-title'>Benchmarking Insights</p>", unsafe_allow_html=True)
                
                # Comparison items
                comp_items = [
                    ('Income', 'monthly_income', '$'),
                    ('Savings', 'monthly_savings', '$'),
                    ('Health Score', 'financial_health_score', ''),
                    ('Credit Score', 'credit_score', '')
                ]
                
                for label, col, unit in comp_items:
                    u_val = user_data[col]
                    p_val = peers_avg[col]
                    diff = (u_val - p_val) / p_val * 100 if p_val != 0 else 0
                    
                    color = "#10b981" if u_val >= p_val else "#ef4444"
                    icon = "🟢" if u_val >= p_val else "🔴"
                    
                    st.markdown(f"""
                    **{label}**: {unit}{u_val:,.1f} vs Peer Avg {unit}{p_val:,.1f} 
                    <span style='color: {color}; font-weight: bold;'>({icon} {diff:+.1f}%)</span>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"**Similar Users Found**: {len(similar_users)}")
                st.dataframe(similar_users[['user_id', 'age', 'employment_type', 'financial_health_score', 'risk_label']], use_container_width=True)

    # ============ SCENARIO SIMULATOR ============
    elif page == "🔮 Scenario Simulator":
        st.header("Financial Scenario Simulator")
        
        st.info("Simulate how changes in your income, expenses, and debt would affect your overall financial health score and risk category.")
        
        # Simulator inputs
        with st.container():
            st.markdown("<p class='card-title'>Simulation Parameters</p>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                sim_income = st.number_input("Monthly Income ($)", 0, 50000, 5000)
                sim_expenses = st.number_input("Monthly Expenses ($)", 0, 50000, 3500)
            with col2:
                sim_savings = st.number_input("Monthly Savings ($)", 0, 50000, 1000)
                sim_investments = st.number_input("Monthly Investments ($)", 0, 50000, 500)
            with col3:
                sim_credit = st.slider("Credit Score", 300, 850, 700)
                sim_debt = st.number_input("Total Debt ($)", 0, 1000000, 50000)
        
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
        # The classification model needs several features. We'll fill in averages for missing ones.
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
        
        # Feature engineering needed
        sim_features['savings_rate'] = sim_features['monthly_savings'] / sim_features['monthly_income'].replace(0, 1)
        sim_features['expense_ratio'] = sim_features['monthly_expenses'] / sim_features['monthly_income'].replace(0, 1)
        sim_features['investment_rate'] = sim_features['monthly_investments'] / sim_features['monthly_income'].replace(0, 1)
        sim_features['debt_to_income'] = sim_features['total_debt'] / (sim_features['monthly_income'] * 12).replace(0, 1)
        sim_features['loan_to_income'] = sim_features['monthly_loan_payments'] / sim_features['monthly_income'].replace(0, 1)
        
        # Ensure correct column order for classifier
        # Use the stored feature names from the model
        sim_features = sim_features[pipeline.classification_model.feature_names]
        
        # Predict risk
        sim_risk_idx = pipeline.classification_model.predict(sim_features)[0]
        # Label encoder mapping
        _, _, le = prepare_classification_data(users_df)
        sim_risk_label = le.inverse_transform([sim_risk_idx])[0]
        
        # Simulation Results
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown("<p class='card-title'>Predicted Health Score</p>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; color: #2563eb;'>{sim_health_score:.1f}</h1>", unsafe_allow_html=True)
                
                # Progress bar for health score
                st.progress(sim_health_score / 100)
                
                st.markdown(f"""
                <div class='sidebar-text' style='text-align: center;'>
                Your simulated profile is in the <b>{sim_risk_label}</b> risk category.
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            with st.container():
                st.markdown("<p class='card-title'>Comparison to Average</p>", unsafe_allow_html=True)
                
                # Comparison metrics
                avg_income = users_df['monthly_income'].mean()
                avg_health = users_df['financial_health_score'].mean()
                
                income_delta = (sim_income - avg_income) / avg_income * 100
                health_delta = sim_health_score - avg_health
                
                st.metric("Income vs Market", f"${sim_income:,.0f}", f"{income_delta:+.1f}%")
                st.metric("Health vs Market", f"{sim_health_score:.1f}", f"{health_delta:+.1f} pts")
        
        # Actionable insights for simulator
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<p class='card-title'>Simulation Insights</p>", unsafe_allow_html=True)
            
            if sim_health_score < 50:
                st.warning("Your simulated profile shows high financial stress. Consider reducing debt and increasing the savings rate.")
            elif sim_health_score < 75:
                st.info("You have a solid foundation. To reach an excellent score, focus on increasing investments and maintaining a high credit score.")
            else:
                st.success("Excellent! This simulated profile represents strong financial health. Keep up the disciplined savings and investment approach.")
                
            st.write("**Top Tip:** Try increasing your 'Monthly Savings' by just $200 in the simulator above to see how it moves your health score!")

    # ============ DATA EXPLORER ============
    elif page == "🔍 Data Explorer":
        st.header("Data Explorer")
        
        st.info("Directly interact with the underlying data and view detailed distributions.")
        
        data_choice = st.selectbox("Select Dataset", ["User Profiles", "Monthly Transactions"])
        
        if data_choice == "User Profiles":
            with st.container():
                st.markdown("<p class='card-title'>User Profiles Filter</p>", unsafe_allow_html=True)
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_income = st.slider("Min Income", 0, 20000, 0)
                with col2:
                    max_income = st.slider("Max Income", 0, 20000, 20000)
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
                st.write(f"Showing {len(filtered)} of {len(users_df)} users")
            
        else:
            with st.container():
                st.markdown("<p class='card-title'>Monthly Transactions Filter</p>", unsafe_allow_html=True)
                
                # Month selector
                month_filter = st.selectbox("Select Month", list(range(1, 13)), format_func=lambda x: f"Month {x}")
                
                filtered = monthly_df[monthly_df['month'] == month_filter]
                
                st.dataframe(filtered.head(100), use_container_width=True)
                st.write(f"Showing {len(filtered)} records for Month {month_filter}")
        
        # Statistics
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<p class='card-title'>Statistical Summary</p>", unsafe_allow_html=True)
            
            if data_choice == "User Profiles":
                st.dataframe(users_df.describe(), use_container_width=True)
            else:
                st.dataframe(monthly_df.describe(), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>💰 AI-Powered Personal Finance Health Advisor | Built with Streamlit</p>
        <p>End-to-End ML Pipeline: Data Generation → Preprocessing → ML Models → Visualization</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()