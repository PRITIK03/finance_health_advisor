"""
User Segmentation Page Module
K-Means clustering analysis for 5 financial profiles.
"""
import streamlit as st
import pandas as pd
from components import UIComponents


def render_user_segmentation(users_df, monthly_df, visualizer):
    """Render the User Segmentation Analysis page."""
    UIComponents.page_header(
        "User Segmentation Analysis",
        "K-Means clustering identifies 5 distinct financial profiles based on behavior and wealth metrics.",
        icon="👥"
    )
    UIComponents.info_box("K-Means clustering identifies 5 distinct financial profiles based on behavior and wealth metrics.")

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
