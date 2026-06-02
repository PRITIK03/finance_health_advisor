"""
FIRE Tracker Page Module
Financial Independence, Retire Early (FIRE) tracker with projection and scenarios.
"""
import streamlit as st
import pandas as pd
from components import UIComponents
from preprocessing import calculate_fire_metrics
import plotly.graph_objects as go


def render_fire_tracker(users_df, monthly_df, visualizer):
    """Render the FIRE Tracker page."""
    UIComponents.page_header(
        "FIRE Tracker",
        "Calculate your path to financial independence and see when you can retire early.",
        icon="🔥"
    )
    UIComponents.info_box("Calculate your path to financial independence and see when you can retire early based on your current savings rate and investments.")

    user_row = UIComponents.user_selector(users_df, key="fire_user")

    with st.container(border=True):
        st.markdown("<p class='card-title'>FIRE Parameters</p>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            safe_withdrawal = st.slider("Safe Withdrawal Rate (%)", 2.0, 6.0, 4.0, 0.1) / 100
        with col2:
            expected_return = st.slider("Expected Return (%)", 3.0, 12.0, 7.0, 0.5) / 100
        with col3:
            inflation_rate = st.slider("Inflation Rate (%)", 0.0, 5.0, 3.0, 0.1) / 100
        with col4:
            current_investments = st.number_input(
                "Current Investments ($)",
                min_value=0,
                value=int(user_row.get('monthly_investments', 0) * 24),
                step=1000
            )

    annual_expenses = float(user_row['monthly_expenses']) * 12
    monthly_contribution = float(user_row['monthly_savings']) + float(user_row['monthly_investments'])
    annual_contribution = monthly_contribution * 12
    real_return = expected_return - inflation_rate

    years_to_fire = 0
    temp_wealth = current_investments
    fire_number = annual_expenses / safe_withdrawal if safe_withdrawal > 0 else float('inf')

    if annual_contribution > 0 and real_return > 0:
        while temp_wealth < fire_number and years_to_fire < 100:
            temp_wealth = (temp_wealth + annual_contribution) * (1 + real_return)
            years_to_fire += 1
    elif annual_contribution > 0:
        years_to_fire = (fire_number - current_investments) / annual_contribution if annual_contribution > 0 else float('inf')

    lean_fire = annual_expenses * 0.7 / safe_withdrawal if safe_withdrawal > 0 else 0
    fat_fire = annual_expenses * 1.5 / safe_withdrawal if safe_withdrawal > 0 else 0
    progress_pct = min(100, (current_investments / fire_number) * 100) if fire_number > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("FIRE Number", f"${fire_number:,.0f}", help="Amount needed to retire with current expenses")
    with col2: st.metric("Years to FIRE", f"{years_to_fire}", help="Estimated years until financial independence")
    with col3: st.metric("Current Progress", f"{progress_pct:.1f}%", help="Percentage of FIRE goal achieved")
    with col4: st.metric("Monthly Investment", f"${monthly_contribution:,.0f}", help="Current monthly savings + investments")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["FIRE Analysis", "Growth Projection", "FIRE Scenarios"])

    with tab1:
        col1, col2 = st.columns([1.5, 2.5])
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Your FIRE Targets</p>", unsafe_allow_html=True)

                lean_progress = min(100, (current_investments / lean_fire) * 100) if lean_fire > 0 else 0
                st.markdown(f"**Lean FIRE:** ${lean_fire:,.0f} (70% of expenses)")
                st.progress(lean_progress / 100, text=f"{lean_progress:.1f}%")
                st.markdown(f"**Regular FIRE:** ${fire_number:,.0f} (100% of expenses)")
                st.progress(progress_pct / 100, text=f"{progress_pct:.1f}%")
                fat_progress = min(100, (current_investments / fat_fire) * 100) if fat_fire > 0 else 0
                st.markdown(f"**Fat FIRE:** ${fat_fire:,.0f} (150% of expenses)")
                st.progress(fat_progress / 100, text=f"{fat_progress:.1f}%")

                st.markdown("---")
                st.markdown(f"**Annual Expenses:** ${annual_expenses:,.0f}")
                st.markdown(f"**Safe Withdrawal:** {safe_withdrawal*100:.1f}% = ${annual_expenses:,.0f}/year")

            with st.container(border=True):
                st.markdown("<p class='card-title'>FIRE Readiness Score</p>", unsafe_allow_html=True)
                st.plotly_chart(
                    visualizer.create_gauge_chart(min(100, progress_pct * 2), "FIRE Readiness", target=100),
                    use_container_width=True
                )

        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>FIRE Milestone Timeline</p>", unsafe_allow_html=True)

                milestones = []
                milestone_names = ["25%", "50%", "75%", "90%", "100% (FIRE)"]
                milestone_values = [fire_number * p for p in [0.25, 0.5, 0.75, 0.9, 1.0]]

                for name, target in zip(milestone_names, milestone_values):
                    if current_investments >= target:
                        milestones.append({"Milestone": name, "Status": "Achieved", "Target": f"${target:,.0f}"})
                    else:
                        remaining = target - current_investments
                        if annual_contribution > 0 and real_return > 0:
                            years_remaining = 0
                            temp = current_investments
                            while temp < target and years_remaining < 100:
                                temp = (temp + annual_contribution) * (1 + real_return)
                                years_remaining += 1
                            milestones.append({"Milestone": name, "Status": f"{years_remaining} years", "Target": f"${target:,.0f}"})
                        else:
                            milestones.append({"Milestone": name, "Status": "Not achievable", "Target": f"${target:,.0f}"})

                st.dataframe(pd.DataFrame(milestones), use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("<p class='card-title'>Savings Rate Impact</p>", unsafe_allow_html=True)

                current_savings_rate = (user_row['monthly_savings'] / user_row['monthly_income'] * 100) if user_row['monthly_income'] > 0 else 0

                if current_savings_rate < 20:
                    st.warning(f"Your savings rate is {current_savings_rate:.1f}%. Increasing to 50%+ could accelerate FIRE.")
                elif current_savings_rate < 50:
                    st.info(f"Good progress! At {current_savings_rate:.1f}% savings rate, you're on the path to FIRE.")
                else:
                    st.success(f"Excellent! Your {current_savings_rate:.1f}% savings rate puts you on the fast track to FIRE!")

    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Net Worth Projection to FIRE</p>", unsafe_allow_html=True)

                years = list(range(min(years_to_fire + 6, 51)))
                wealth_values = []
                temp = current_investments
                for _ in years:
                    wealth_values.append(temp)
                    temp = (temp + annual_contribution) * (1 + real_return)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years, y=wealth_values,
                    mode='lines+markers',
                    name='Projected Net Worth',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=6)
                ))

                fig.add_hline(y=fire_number, line_dash="dash", line_color="#ef4444",
                             annotation_text="FIRE Number", annotation_position="top right")
                fig.add_hline(y=lean_fire, line_dash="dot", line_color="#f59e0b",
                             annotation_text="Lean FIRE", annotation_position="bottom right")

                UIComponents.plotly_defaults(fig, height=350)
                fig.update_layout(
                    xaxis_title="Years from Now",
                    yaxis_title="Net Worth ($)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Contribution Analysis</p>", unsafe_allow_html=True)

                st.metric("Current Investments", f"${current_investments:,.0f}")
                st.metric("Annual Contribution", f"${annual_contribution:,.0f}")
                st.metric("Real Return Rate", f"{real_return*100:.1f}%")

                st.markdown("---")
                st.markdown("**Quick Adjustments:**")

                extra_100 = 100 * 12
                temp_with_extra = current_investments
                years_with_extra = 0
                while temp_with_extra < fire_number and years_with_extra < 100:
                    temp_with_extra = (temp_with_extra + annual_contribution + extra_100) * (1 + real_return)
                    years_with_extra += 1

                time_saved = years_to_fire - years_with_extra
                if time_saved > 0:
                    st.success(f"+${extra_100/12:.0f}/month saves {time_saved} years!")

    with tab3:
        with st.container(border=True):
            st.markdown("<p class='card-title'>Compare FIRE Scenarios</p>", unsafe_allow_html=True)

            scenarios = []
            for withdrawal in [0.03, 0.04, 0.05]:
                for return_rate in [0.05, 0.07, 0.09]:
                    real = return_rate - inflation_rate
                    fire_num = annual_expenses / withdrawal if withdrawal > 0 else float('inf')

                    years = 0
                    temp = current_investments
                    annual = annual_contribution
                    while temp < fire_num and years < 100:
                        temp = (temp + annual) * (1 + real)
                        years += 1

                    scenarios.append({
                        "Withdrawal Rate": f"{withdrawal*100:.0f}%",
                        "Return Rate": f"{return_rate*100:.0f}%",
                        "FIRE Number": f"${fire_num:,.0f}",
                        "Years to FIRE": years if years < 100 else "100+",
                        "Monthly Income": f"${fire_num * withdrawal / 12:,.0f}"
                    })

            st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

            st.info("Key Insight: Lower withdrawal rates require more savings but provide more safety. Higher returns accelerate your timeline but come with more volatility.")
