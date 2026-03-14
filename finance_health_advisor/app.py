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

from data_generator import generate_full_dataset
from preprocessing import FinancialDataPreprocessor, prepare_classification_data
from models import train_all_models
from visualizations import FinancialVisualizer, generate_summary_statistics


@st.cache_data
def load_data():
    """Load or generate data."""
    users_df, monthly_df = generate_full_dataset(10000)
    return users_df, monthly_df


@st.cache_resource
def train_models(users_df, monthly_df):
    """Train ML models."""
    results = train_all_models(users_df, monthly_df)
    return results


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
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 20px;
    }
    
    .custom-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 15px;
    }
    
    .sidebar-text {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    .highlight {
        color: #3b82f6;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #64748b;
        font-weight: 600;
        font-size: 1rem;
    }

    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom-color: #3b82f6 !important;
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
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Generating synthetic financial data..."):
        users_df, monthly_df = load_data()
    
    # Train models
    with st.spinner("Training ML models..."):
        results = train_all_models(users_df, monthly_df)
    
    # Preprocess data
    preprocessor = FinancialDataPreprocessor()
    users_processed = preprocessor.preprocess_users(users_df)
    monthly_processed = preprocessor.preprocess_monthly(monthly_df)
    
    # Visualizations
    visualizer = FinancialVisualizer(users_df, monthly_df)
    
    # Navigation
    page = st.sidebar.radio(
        "Select Section",
        ["📊 Dashboard Overview", "👥 User Segmentation", "🎯 Risk Prediction", 
         "📈 Forecasting", "🚨 Anomaly Detection", "🔍 Data Explorer"]
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
                    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                    st.plotly_chart(visualizer.create_risk_analysis_plot(), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                with st.container():
                    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                    st.plotly_chart(visualizer.create_spending_breakdown_chart(), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with st.container():
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_time_series_plot(), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                    st.plotly_chart(visualizer.create_income_expense_scatter(), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                with st.container():
                    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                    st.plotly_chart(visualizer.create_employment_analysis(), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            with st.container():
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_correlation_heatmap(), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
        with tab3:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Recent Financial Snapshot")
            st.dataframe(users_df[['user_id', 'age', 'employment_type', 'monthly_income', 'monthly_expenses', 'financial_health_score']].head(20), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ============ USER SEGMENTATION ============
    elif page == "👥 User Segmentation":
        st.header("User Segmentation Analysis")
        
        st.info("K-Means clustering identifies 5 distinct financial profiles based on behavior and wealth metrics.")
        
        tab1, tab2 = st.tabs(["📊 Cluster Overview", "📋 Detailed Statistics"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.plotly_chart(visualizer.create_cluster_analysis_plot(), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.markdown("<p class='card-title'>Cluster Distribution</p>", unsafe_allow_html=True)
                cluster_counts = users_df['cluster'].value_counts().sort_index()
                for i, count in cluster_counts.items():
                    st.write(f"**Cluster {i}:** {count} users")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Cluster Profiles")
            cluster_stats = users_df.groupby('cluster').agg({
                'monthly_income': 'mean',
                'monthly_expenses': 'mean',
                'monthly_savings': 'mean',
                'credit_score': 'mean',
                'financial_health_score': 'mean'
            }).round(2)
            st.dataframe(cluster_stats, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.subheader("Profile Descriptions")
            cols = st.columns(3)
            for i, cluster_id in enumerate(sorted(users_df['cluster'].unique())):
                with cols[i % 3]:
                    cluster_data = users_df[users_df['cluster'] == cluster_id]
                    st.markdown(f"""
                    <div class='custom-card'>
                        <p class='card-title'>Profile {cluster_id}</p>
                        Avg Health: <span class='highlight'>{cluster_data['financial_health_score'].mean():.1f}</span><br>
                        Avg Income: ${cluster_data['monthly_income'].mean():,.0f}<br>
                        Users: {len(cluster_data)}
                    </div>
                    """, unsafe_allow_html=True)
    
    # ============ RISK PREDICTION ============
    elif page == "🎯 Risk Prediction":
        st.header("Financial Risk Analysis")
        
        st.info("Random Forest classifier predicts financial vulnerability by analyzing user profiles and historical data.")
        
        # Model metrics
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
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
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
                fig.update_layout(template="plotly_white", margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.markdown("<p class='card-title'>Risk Insights</p>", unsafe_allow_html=True)
                st.write("The model classifies users into 5 risk levels. Most users fall into the **Medium** to **Low** risk categories.")
                st.write("Key drivers for risk include **Credit Score**, **Monthly Savings**, and **Total Debt**.")
                st.markdown("</div>", unsafe_allow_html=True)
                
        with tab2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
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
            fig.update_layout(template="plotly_white", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
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
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tab2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)
    
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
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.plotly_chart(visualizer.create_anomaly_analysis_plot(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tab2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<p class='card-title'>Sample of Flagged Transactions</p>", unsafe_allow_html=True)
            
            anomalies = monthly_df[monthly_df['ml_anomaly'] == 1]
            st.dataframe(
                anomalies[['user_id', 'month', 'income', 'expenses', 'savings', 'Entertainment', 'Shopping']].head(20),
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ============ DATA EXPLORER ============
    elif page == "🔍 Data Explorer":
        st.header("Data Explorer")
        
        st.info("Directly interact with the underlying data and view detailed distributions.")
        
        data_choice = st.selectbox("Select Dataset", ["User Profiles", "Monthly Transactions"])
        
        if data_choice == "User Profiles":
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<p class='card-title'>Monthly Transactions Filter</p>", unsafe_allow_html=True)
            
            # Month selector
            month_filter = st.selectbox("Select Month", list(range(1, 13)), format_func=lambda x: f"Month {x}")
            
            filtered = monthly_df[monthly_df['month'] == month_filter]
            
            st.dataframe(filtered.head(100), use_container_width=True)
            st.write(f"Showing {len(filtered)} records for Month {month_filter}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Statistics
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>Statistical Summary</p>", unsafe_allow_html=True)
        
        if data_choice == "User Profiles":
            st.dataframe(users_df.describe(), use_container_width=True)
        else:
            st.dataframe(monthly_df.describe(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
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