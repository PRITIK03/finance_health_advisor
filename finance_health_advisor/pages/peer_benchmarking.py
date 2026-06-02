"""
Peer Benchmarking Page Module
KNN-based peer comparison.
"""
import streamlit as st
from components import UIComponents


def render_peer_benchmarking(users_df, monthly_df, visualizer):
    """Render the Peer Benchmarking page."""
    UIComponents.page_header(
        "Peer Benchmarking",
        "Compare yourself to similar financial profiles using KNN.",
        icon="👥"
    )
    UIComponents.info_box("Compare yourself to similar financial profiles using KNN.", icon="📊")

    user_id = st.selectbox("Select User", users_df['user_id'].unique())
    user_row = users_df[users_df['user_id'] == user_id].iloc[0]

    st.plotly_chart(visualizer.create_gauge_chart(user_row['financial_health_score'], f"User {user_id} Health"), use_container_width=True)
    st.caption("Peer comparison radar coming from KNN model (see models.py).")
