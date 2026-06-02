import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

st.set_page_config(
    page_title="Finance Health Advisor",
    page_icon="≡ƒÆ░",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_full_dataset, calculate_financial_health_score
from preprocessing import FinancialDataPreprocessor, prepare_classification_data
from models import train_all_models
from visualizations import FinancialVisualizer
from recommendations import RecommendationsEngine
from styles import ThemeManager
from pages.router import PAGE_ROUTES

TOTAL_USERS = 10000


@st.cache_data
def load_data(offset: int = 0, limit: int = None):
    users_df, monthly_df = generate_full_dataset(total_users=TOTAL_USERS, offset=offset, limit=limit)
    return users_df, monthly_df


@st.cache_resource
def train_models(users_df, monthly_df):
    results, pipeline = train_all_models(users_df, monthly_df)
    return results, pipeline


st.markdown(f"""
<div class='main-header'>
    <h1 style='color: white; margin: 0;'>Finance Health Advisor</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0;'>AI-Powered Financial Insights & Behavioral Analysis</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 1
if 'page_size' not in st.session_state:
    st.session_state['page_size'] = 100

page_size_options = [50, 100, 200, 500]
st.session_state['page_size'] = st.sidebar.selectbox("Users per page", page_size_options, index=1)
total_pages = (TOTAL_USERS // st.session_state['page_size']) + (1 if TOTAL_USERS % st.session_state['page_size'] > 0 else 0)
offset = (st.session_state['page_number'] - 1) * st.session_state['page_size']
limit = st.session_state['page_size']

with st.spinner(f"Generating synthetic financial data for page {st.session_state['page_number']}..."):
    users_df, monthly_df = load_data(offset=offset, limit=limit)
with st.spinner("Training ML models..."):
    results, pipeline = train_models(users_df, monthly_df)
preprocessor = FinancialDataPreprocessor()
users_processed = preprocessor.preprocess_users(users_df)
monthly_processed = preprocessor.preprocess_monthly(monthly_df)
visualizer = FinancialVisualizer(users_df, monthly_df)
recommendations_engine = RecommendationsEngine(users_df, monthly_df)

with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding-bottom: 10px;'>
        <div style='font-size: 3rem;'>≡ƒÆ░</div>
        <h2 style='margin-top: 5px; color: #0f172a;'>Menu</h2>
    </div>
    """, unsafe_allow_html=True)

    theme = st.toggle("≡ƒîÖ Dark Mode", value=False)
    ThemeManager.apply_theme(theme)

    page = st.radio(
        "Go to section:",
        list(PAGE_ROUTES.keys()),
        label_visibility="collapsed"
    )

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

    st.markdown("---")
    st.subheader("System Insights")
    avg_health = users_df['financial_health_score'].mean()
    health_status = "Good" if avg_health > 70 else "Fair" if avg_health > 50 else "Poor"
    st.plotly_chart(visualizer.create_mini_bullet(avg_health, 85, "Avg Health Index", color="#10b981"), use_container_width=True)

    st.markdown("---")
    st.subheader("≡ƒÅå Achievements")
    col1_ach, col2_ach = st.columns(2)
    with col1_ach:
        st.markdown("≡ƒ¢í∩╕Å **Debt-Free**")
        st.markdown("≡ƒöÑ **FIRE Starter**")
    with col2_ach:
        st.markdown("≡ƒîƒ **Elite Credit**")
        st.markdown("≡ƒÅª **Wealth Builder**")

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

    st.markdown("---")
    st.subheader("Export Center")

context = {
    'users_df': users_df,
    'monthly_df': monthly_df,
    'recommendations_engine': recommendations_engine,
    'visualizer': visualizer,
    'results': results,
    'pipeline': pipeline,
}

if page in PAGE_ROUTES:
    PAGE_ROUTES[page](**context)
else:
    st.warning("≡ƒÜº This section is under active development.")
