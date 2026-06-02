"""
Data Explorer Page Module
Raw data explorer with download links.
"""
import streamlit as st
from components import UIComponents


def render_data_explorer(users_df, monthly_df):
    """Render the Data Explorer page."""
    UIComponents.page_header(
        "Raw Data Explorer",
        "Browse and export the underlying financial dataset.",
        icon="🔍"
    )
    UIComponents.info_box("Browse and export the underlying financial dataset.", icon="📁")

    tab1, tab2 = st.tabs(["Users", "Monthly Transactions"])

    with tab1:
        st.dataframe(users_df, use_container_width=True, height=500)
        st.download_button("Download Users CSV", users_df.to_csv(index=False), "users.csv", "text/csv")

    with tab2:
        st.dataframe(monthly_df, use_container_width=True, height=500)
        st.download_button("Download Monthly CSV", monthly_df.to_csv(index=False), "monthly.csv", "text/csv")
