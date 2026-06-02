"""
Wealth Projection Page Module
Interactive wealth growth projection with compound interest calculator.
"""
import streamlit as st
import pandas as pd
from components import UIComponents
from preprocessing import calculate_wealth_projection
import plotly.graph_objects as go


def render_wealth_projection(users_df, monthly_df, visualizer):
    """Render the Wealth Projection Simulator page."""
    UIComponents.page_header(
        "Wealth Projection Simulator",
        "Project your financial future and see how your investments and savings can grow over time with compound interest.",
        icon="🚀"
    )

    UIComponents.info_box("Project your financial future! See how your investments and savings can grow over time with compound interest.")

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
            ) / 100
            projection_years = st.slider(
                "Projection Years",
                min_value=1,
                max_value=50,
                value=20,
                step=1
            )

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

            growth_pct = ((projection_results['final_wealth'] - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
            st.plotly_chart(
                visualizer.create_gauge_chart(
                    min(1000, growth_pct),
                    "Wealth Growth (%)",
                    target=100
                ),
                use_container_width=True
            )

    with col2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Wealth Growth Over Time</p>", unsafe_allow_html=True)

            fig = go.Figure()
            months = list(range(len(projection_results['wealth_history'])))

            fig.add_trace(go.Scatter(
                x=[m / 12 for m in months],
                y=projection_results['wealth_history'],
                mode='lines',
                name='Projected Wealth',
                line=dict(color='#10b981', width=3)
            ))

            UIComponents.plotly_defaults(fig, height=350)
            fig.update_layout(
                xaxis_title="Years",
                yaxis_title="Wealth ($)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
