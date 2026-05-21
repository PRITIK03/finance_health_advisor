"""
Monte Carlo Wealth Simulator Page
Stochastic projections with volatility for realistic outcome ranges.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def run_monte_carlo(initial, monthly, annual_return, volatility, years, n_sims=1000):
    """Run Monte Carlo simulation for wealth projection."""
    months = years * 12
    monthly_return = (1 + annual_return) ** (1/12) - 1
    monthly_vol = volatility / np.sqrt(12)

    results = np.zeros((n_sims, months + 1))
    results[:, 0] = initial

    for sim in range(n_sims):
        wealth = initial
        for m in range(1, months + 1):
            # Random monthly return
            rand_return = np.random.normal(monthly_return, monthly_vol)
            wealth = (wealth + monthly) * (1 + rand_return)
            results[sim, m] = wealth

    return results


def render_monte_carlo_simulator(users_df, monthly_df):
    st.header("📉 Monte Carlo Wealth Simulator")
    st.info("Run thousands of possible futures with market volatility. See the range of outcomes instead of a single straight line.")
    st.caption("Note: Uses currently loaded users from the active data page.")

    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Starting Capital ($)", min_value=0, value=25000, step=1000)
        monthly_contrib = st.number_input("Monthly Contribution ($)", min_value=0, value=800, step=50)
        years = st.slider("Projection Horizon (Years)", 5, 40, 25)
    with col2:
        expected_return = st.slider("Expected Annual Return (%)", 3.0, 12.0, 7.0, 0.5) / 100
        volatility = st.slider("Annual Volatility (%)", 5.0, 25.0, 15.0, 0.5) / 100
        n_sims = st.select_slider("Number of Simulations", options=[500, 1000, 2000, 5000], value=1000)

    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            sim_results = run_monte_carlo(
                initial_capital, monthly_contrib, expected_return, volatility, years, n_sims
            )

        final_wealths = sim_results[:, -1]

        median = np.median(final_wealths)
        p10 = np.percentile(final_wealths, 10)
        p90 = np.percentile(final_wealths, 90)
        prob_below_initial = (final_wealths < initial_capital).mean() * 100

        st.markdown("### Simulation Results")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Median Outcome", f"${median:,.0f}")
        with m2:
            st.metric("10th Percentile (Bad Case)", f"${p10:,.0f}")
        with m3:
            st.metric("90th Percentile (Great Case)", f"${p90:,.0f}")
        with m4:
            st.metric("Chance of Losing Money", f"{prob_below_initial:.1f}%")

        # Distribution histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=final_wealths,
            nbinsx=60,
            name="Final Wealth Distribution",
            marker_color="#3b82f6",
            opacity=0.75
        ))
        fig.add_vline(x=median, line_dash="dash", line_color="#10b981", annotation_text="Median")
        fig.add_vline(x=p10, line_dash="dot", line_color="#ef4444", annotation_text="10th %")
        fig.add_vline(x=p90, line_dash="dot", line_color="#10b981", annotation_text="90th %")

        fig.update_layout(
            title="Distribution of Possible Outcomes",
            xaxis_title="Final Wealth ($)",
            yaxis_title="Number of Simulations",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Key Insight**")
        if prob_below_initial > 15:
            st.warning("High volatility + low contribution rate creates significant downside risk.")
        else:
            st.success("With consistent contributions, the probability of growing your wealth is very high even in bad markets.")

        st.caption(f"Assumptions: Lognormal returns, constant monthly contributions, no fees or taxes modeled. {n_sims:,} paths simulated.")
