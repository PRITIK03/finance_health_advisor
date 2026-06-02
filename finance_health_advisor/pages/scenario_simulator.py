"""
Scenario Simulator Page Module
Custom what-if Monte Carlo scenario modeling.
"""
import streamlit as st
from components import UIComponents


def render_scenario_simulator():
    """Render the Scenario Simulator page."""
    UIComponents.page_header(
        "What-If Scenario Simulator",
        "Run custom Monte Carlo simulations on your financial future.",
        icon="🔮"
    )
    UIComponents.info_box("Run custom Monte Carlo simulations on your financial future.", icon="🎲")

    st.write("**Coming soon:** Full Monte Carlo wealth simulator with adjustable volatility, sequence-of-returns risk, and custom life events.")
