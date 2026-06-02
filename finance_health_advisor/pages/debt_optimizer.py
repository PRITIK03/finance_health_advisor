"""
Debt Optimizer Page Module
Smart debt payoff strategies with Avalanche vs Snowball comparison.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components import UIComponents
from preprocessing import calculate_debt_paydown
from datetime import datetime, timedelta


def render_debt_optimizer(users_df, monthly_df, visualizer):
    """Render the Smart Debt Optimizer page."""
    UIComponents.page_header(
        "Smart Debt Optimizer",
        "Compare debt payoff strategies and find the optimal approach to eliminate your debt faster while saving on interest payments.",
        icon="💸"
    )
    UIComponents.info_box("Compare debt payoff strategies and find the optimal approach to eliminate your debt faster while saving on interest payments.")

    user_row = UIComponents.user_selector(users_df, key="debt_user")

    default_debts = [
        {"name": "Credit Card", "balance": float(user_row['total_debt']) * 0.4 if user_row['total_debt'] > 0 else 5000,
         "interest_rate": 19.99, "min_payment": 150},
        {"name": "Car Loan", "balance": float(user_row['total_debt']) * 0.35 if user_row['total_debt'] > 0 else 8000,
         "interest_rate": 5.99, "min_payment": 300},
        {"name": "Student Loan", "balance": float(user_row['total_debt']) * 0.25 if user_row['total_debt'] > 0 else 6000,
         "interest_rate": 4.5, "min_payment": 200},
    ]

    with st.container(border=True):
        st.markdown("<p class='card-title'>Debt Configuration</p>", unsafe_allow_html=True)

        debts = []
        for i, default in enumerate(default_debts):
            with st.expander(f"Debt {i+1}: {default['name']}", expanded=i==0):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    name = st.text_input(f"Name {i}", value=default['name'], key=f"debt_name_{i}")
                with col2:
                    balance = st.number_input(f"Balance ${i}", min_value=0.0, value=float(default['balance']), step=100.0, key=f"debt_bal_{i}")
                with col3:
                    rate = st.number_input(f"APR % {i}", min_value=0.0, max_value=50.0, value=float(default['interest_rate']), step=0.1, key=f"debt_rate_{i}")
                with col4:
                    min_pay = st.number_input(f"Min Pay ${i}", min_value=0.0, value=float(default['min_payment']), step=10.0, key=f"debt_min_{i}")

                if balance > 0:
                    debts.append({"name": name, "balance": balance, "interest_rate": rate, "min_payment": min_pay})

        col1, col2 = st.columns(2)
        with col1:
            extra_payment = st.number_input("Extra Monthly Payment ($)", min_value=0.0, value=float(user_row['monthly_savings'] * 0.5), step=50.0,
                                           help="Additional amount to put toward debt payoff each month")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            optimize_button = st.button("Calculate Optimal Strategy", use_container_width=True, type="primary")

    if debts and optimize_button:
        results = calculate_debt_paydown(debts, extra_payment)
        avalanche = results['avalanche']
        snowball = results['snowball']

        total_debt = sum(d['balance'] for d in debts)
        min_monthly = sum(d['min_payment'] for d in debts)

        st.markdown("<br>", unsafe_allow_html=True)

        UIComponents.metric_row([
            {"label": "Total Debt", "value": f"${total_debt:,.0f}"},
            {"label": "Min. Monthly Payment", "value": f"${min_monthly:,.0f}"},
            {"label": "Extra Payment", "value": f"${extra_payment:,.0f}"},
            {"label": "Total Monthly", "value": f"${min_monthly + extra_payment:,.0f}"},
        ], columns=4)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Avalanche Method (Highest Interest First)</p>", unsafe_allow_html=True)
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.metric("Payoff Time", f"{avalanche['months']} months")
                    st.metric("Total Interest", f"${avalanche['total_interest']:,.0f}")
                with col_a2:
                    st.metric("Years to Debt-Free", f"{avalanche['months']/12:.1f}")
                    st.metric("Avg Monthly Cost", f"${(total_debt + avalanche['total_interest']) / avalanche['months']:,.0f}")
                st.markdown("---")
                st.markdown("**Priority Order:** Pay minimums on all, put extra toward highest interest debt first.")
                st.success("Best for: Saving the most money on interest")

        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Snowball Method (Smallest Balance First)</p>", unsafe_allow_html=True)
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Payoff Time", f"{snowball['months']} months")
                    st.metric("Total Interest", f"${snowball['total_interest']:,.0f}")
                with col_s2:
                    st.metric("Years to Debt-Free", f"{snowball['months']/12:.1f}")
                    st.metric("Avg Monthly Cost", f"${(total_debt + snowball['total_interest']) / snowball['months']:,.0f}")
                st.markdown("---")
                st.markdown("**Priority Order:** Pay minimums on all, put extra toward smallest balance first.")
                st.info("Best for: Quick wins and motivation")

        st.markdown("<br>", unsafe_allow_html=True)

        interest_saved = snowball['total_interest'] - avalanche['total_interest']
        months_diff = snowball['months'] - avalanche['months']

        with st.container(border=True):
            st.markdown("<p class='card-title'>AI Recommendation</p>", unsafe_allow_html=True)

            if interest_saved > 100:
                st.success(f"Recommended: Avalanche Method - You'll save ${interest_saved:,.0f} in interest compared to Snowball!")
            elif months_diff > 3:
                st.info(f"Either method works, but Avalanche saves ${interest_saved:,.0f} and pays off {months_diff} months faster.")
            else:
                st.success("Both strategies are similar for your debt profile. Choose based on what motivates you more!")

            st.markdown("---")
            debt_free_date = datetime.now() + timedelta(days=avalanche['months'] * 30)
            st.markdown(f"Estimated Debt-Free Date: {debt_free_date.strftime('%B %Y')} (using Avalanche)")

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Balance Over Time", "Debt Breakdown"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>Avalanche Payoff Timeline</p>", unsafe_allow_html=True)
                    if avalanche['history']:
                        months = list(range(len(avalanche['history'])))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=months, y=avalanche['history'],
                            mode='lines', fill='tozeroy',
                            name='Remaining Balance',
                            line=dict(color='#3b82f6', width=2),
                            fillcolor='rgba(59, 130, 246, 0.2)'
                        ))
                        UIComponents.plotly_defaults(fig, height=280)
                        fig.update_layout(xaxis_title="Months", yaxis_title="Remaining Balance ($)")
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                with st.container(border=True):
                    st.markdown("<p class='card-title'>Snowball Payoff Timeline</p>", unsafe_allow_html=True)
                    if snowball['history']:
                        months = list(range(len(snowball['history'])))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=months, y=snowball['history'],
                            mode='lines', fill='tozeroy',
                            name='Remaining Balance',
                            line=dict(color='#10b981', width=2),
                            fillcolor='rgba(16, 185, 129, 0.2)'
                        ))
                        UIComponents.plotly_defaults(fig, height=280)
                        fig.update_layout(xaxis_title="Months", yaxis_title="Remaining Balance ($)")
                        st.plotly_chart(fig, use_container_width=True)

        with tab2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>Your Debt Portfolio</p>", unsafe_allow_html=True)
                debt_df = pd.DataFrame(debts)
                debt_df['Annual Interest'] = debt_df['balance'] * (debt_df['interest_rate'] / 100)
                debt_df['Cost per Month'] = debt_df['Annual Interest'] / 12
                st.dataframe(
                    debt_df[['name', 'balance', 'interest_rate', 'min_payment', 'Cost per Month']]
                    .style.format({
                        'balance': '${:,.0f}',
                        'interest_rate': '{:.2f}%',
                        'min_payment': '${:,.0f}',
                        'Cost per Month': '${:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                total_annual_interest = debt_df['Annual Interest'].sum()
                st.markdown(f"**Total Annual Interest Cost:** ${total_annual_interest:,.0f} (${total_annual_interest/12:,.0f}/month)")
    else:
        st.info("Configure your debts above and click 'Calculate Optimal Strategy' to see your personalized payoff plan.")
