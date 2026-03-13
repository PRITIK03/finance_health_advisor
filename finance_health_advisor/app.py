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
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("💰 AI-Powered Personal Finance Health Advisor")
    st.markdown("### End-to-End Machine Learning Financial Analysis System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
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
        st.header("📊 Dashboard Overview")
        
        # Summary Statistics
        stats = generate_summary_statistics(users_df, monthly_df)
        
        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{stats['total_users']:,}")
        with col2:
            st.metric("Avg Monthly Income", f"${stats['avg_income']:,.0f}")
        with col3:
            st.metric("Avg Savings", f"${stats['avg_savings']:,.0f}")
        with col4:
            st.metric("Avg Health Score", f"{stats['avg_health_score']:.1f}/100")
        
        st.markdown("---")
        
        # Row 1: Risk Analysis & Correlation
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(visualizer.create_risk_analysis_plot(), use_container_width=True)
        
        with col2:
            st.plotly_chart(visualizer.create_income_expense_scatter(), use_container_width=True)
        
        st.markdown("---")
        
        # Row 2: Spending & Time Series
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(visualizer.create_spending_breakdown_chart(), use_container_width=True)
        
        with col2:
            st.plotly_chart(visualizer.create_time_series_plot(), use_container_width=True)
        
        st.markdown("---")
        
        # Row 3: Employment Analysis
        st.plotly_chart(visualizer.create_employment_analysis(), use_container_width=True)
    
    # ============ USER SEGMENTATION ============
    elif page == "👥 User Segmentation":
        st.header("👥 User Segmentation (K-Means Clustering)")
        
        st.info("The system uses K-Means clustering to segment users into 5 financial health tiers based on their income, expenses, savings, credit score, and debt.")
        
        # Cluster distribution
        cluster_counts = users_df['cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=cluster_counts.index, 
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Users'},
                title='User Distribution by Cluster',
                color=cluster_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cluster statistics
            cluster_stats = users_df.groupby('cluster').agg({
                'monthly_income': 'mean',
                'monthly_expenses': 'mean',
                'monthly_savings': 'mean',
                'credit_score': 'mean',
                'financial_health_score': 'mean'
            }).round(2)
            
            st.dataframe(cluster_stats, use_container_width=True)
        
        st.markdown("---")
        
        # Cluster characteristics
        st.subheader("Cluster Characteristics")
        
        for cluster_id in sorted(users_df['cluster'].unique()):
            cluster_data = users_df[users_df['cluster'] == cluster_id]
            avg_score = cluster_data['financial_health_score'].mean()
            
            st.markdown(f"""
            **{cluster_id}:** {len(cluster_data)} users | Avg Income: ${cluster_data['monthly_income'].mean():,.0f} | 
            Avg Savings: ${cluster_data['monthly_savings'].mean():,.0f} | Avg Health Score: {avg_score:.1f}
            """)
    
    # ============ RISK PREDICTION ============
    elif page == "🎯 Risk Prediction":
        st.header("🎯 Financial Risk Prediction (Random Forest)")
        
        st.info("The Random Forest classifier predicts financial risk categories based on user characteristics.")
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Accuracy", f"{results['classification']['train_metrics']['accuracy']:.2%}")
        with col2:
            st.metric("Test Accuracy", f"{results['classification']['test_metrics']['accuracy']:.2%}")
        with col3:
            st.metric("CV Score", f"{results['classification']['test_metrics']['cv_mean']:.4f} ± {results['classification']['test_metrics']['cv_std']:.4f}")
        
        st.markdown("---")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        X_class, y_class, le = prepare_classification_data(users_df)
        
        # Train a quick model to get feature importance
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
            title='Feature Importance for Risk Prediction',
            color='importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Risk distribution
        st.subheader("Risk Category Distribution")
        
        risk_counts = users_df['risk_label'].value_counts()
        colors = {'Very Low': '#27ae60', 'Low': '#2ecc71', 'Medium': '#f1c40f', 
                 'High': '#e67e22', 'Very High': '#e74c3c'}
        
        fig = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            title='Risk Category Distribution',
            color=risk_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ============ FORECASTING ============
    elif page == "📈 Forecasting":
        st.header("📈 Savings Forecasting (Gradient Boosting)")
        
        st.info("The Gradient Boosting regressor forecasts monthly savings based on user profile and time-based features.")
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score (Test)", f"{results['forecasting']['test_metrics']['r2']:.4f}")
        with col2:
            st.metric("RMSE (Test)", f"${results['forecasting']['test_metrics']['rmse']:.2f}")
        with col3:
            st.metric("MAE (Test)", f"${results['forecasting']['test_metrics']['mae']:.2f}")
        
        st.markdown("---")
        
        # Monthly trends
        st.subheader("Monthly Savings Trends")
        
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
            line=dict(color='#3498db', width=3)
        ))
        
        # Add trend line
        z = np.polyfit(monthly_avg['month'], monthly_avg['savings'], 2)
        p = np.poly1d(z)
        x_trend = np.arange(1, 13)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend',
            line=dict(color='#e74c3c', dash='dash')
        ))
        
        fig.update_layout(
            title='Monthly Average Savings with Trend',
            xaxis_title='Month',
            yaxis_title='Savings ($)',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Savings by user segment
        st.subheader("Savings Patterns by Risk Category")
        
        savings_by_risk = monthly_df.groupby('risk_label')['savings'].mean()
        
        fig = px.bar(
            x=savings_by_risk.index,
            y=savings_by_risk.values,
            labels={'x': 'Risk Category', 'y': 'Average Savings ($)'},
            title='Average Savings by Risk Category',
            color=savings_by_risk.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ============ ANOMALY DETECTION ============
    elif page == "🚨 Anomaly Detection":
        st.header("🚨 Anomaly Detection (Isolation Forest)")
        
        st.info("The Isolation Forest algorithm detects unusual spending patterns that may indicate fraud, errors, or unusual financial behavior.")
        
        # Anomaly statistics
        anomaly_count = monthly_df['ml_anomaly'].sum()
        anomaly_pct = anomaly_count / len(monthly_df) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Anomalies Detected", f"{anomaly_count}")
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")
        
        st.markdown("---")
        
        # Anomaly scatter plot
        anomalies = monthly_df[monthly_df['ml_anomaly'] == 1]
        normal = monthly_df[monthly_df['ml_anomaly'] == 0].sample(min(1000, len(monthly_df[monthly_df['ml_anomaly'] == 0])))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=normal['expenses'],
            y=normal['savings'],
            mode='markers',
            name='Normal',
            marker=dict(color='#3498db', size=5, opacity=0.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=anomalies['expenses'],
            y=anomalies['savings'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#e74c3c', size=10, symbol='x')
        ))
        
        fig.update_layout(
            title='Anomaly Detection: Expenses vs Savings',
            xaxis_title='Expenses ($)',
            yaxis_title='Savings ($)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Anomaly examples
        st.subheader("Sample Anomalies")
        
        st.dataframe(
            anomalies[['user_id', 'month', 'income', 'expenses', 'savings', 'Entertainment', 'Shopping']].head(10),
            use_container_width=True
        )
    
    # ============ DATA EXPLORER ============
    elif page == "🔍 Data Explorer":
        st.header("🔍 Data Explorer")
        
        st.info("Explore the raw data and statistics in detail.")
        
        # Data selector
        data_choice = st.selectbox("Select Dataset", ["User Profiles", "Monthly Transactions"])
        
        if data_choice == "User Profiles":
            st.subheader("User Profile Data")
            
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
            st.subheader("Monthly Transaction Data")
            
            # Month selector
            month_filter = st.selectbox("Select Month", list(range(1, 13)), format_func=lambda x: f"Month {x}")
            
            filtered = monthly_df[monthly_df['month'] == month_filter]
            
            st.dataframe(filtered.head(100), use_container_width=True)
            st.write(f"Showing {len(filtered)} records for Month {month_filter}")
        
        # Statistics
        st.markdown("---")
        st.subheader("Statistical Summary")
        
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