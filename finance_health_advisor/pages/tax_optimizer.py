"""
Tax Optimizer Page
Interactive tax-advantaged contribution planner with estimated savings.
"""
import streamlit as st


def render_tax_optimizer(users_df, monthly_df, recommendations_engine):
    st.header("💼 Tax Optimization Calculator")
    st.info("Maximize your tax-advantaged accounts. See exactly how much you could save this year.")

    user_id = st.selectbox(
        "Select User Profile",
        users_df['user_id'].unique(),
        format_func=lambda x: f"User {x}"
    )
    user_row = users_df[users_df['user_id'] == user_id].iloc[0].to_dict()

    # Get base tax profile
    tax_profile = recommendations_engine.get_tax_profile(user_row)

    st.markdown("### Your Current Tax Situation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Est. Annual Income", f"${tax_profile['annual_income']:,.0f}")
    with c2:
        st.metric("Marginal Tax Rate", f"{tax_profile['marginal_rate']*100:.0f}%")
    with c3:
        st.metric("Tax Bracket", tax_profile['tax_bracket'])

    st.markdown("### Current Contributions (Editable)")
    col1, col2, col3 = st.columns(3)

    with col1:
        current_401k = st.number_input(
            "401(k) Contribution (Annual)", 
            min_value=0, 
            max_value=23000, 
            value=int(tax_profile['current_401k']), 
            step=500
        )
    with col2:
        current_hsa = st.number_input(
            "HSA Contribution (Annual)", 
            min_value=0, 
            max_value=4150 if user_row.get('age', 30) < 55 else 8300, 
            value=int(tax_profile['current_hsa']), 
            step=250
        )
    with col3:
        current_ira = st.number_input(
            "IRA Contribution (Annual)", 
            min_value=0, 
            max_value=7000 if user_row.get('age', 30) < 50 else 8000, 
            value=int(tax_profile['current_ira']), 
            step=500
        )

    # Recompute with new values (simplified override)
    modified_user = user_row.copy()
    modified_user['monthly_investments'] = (current_401k + current_hsa + current_ira) / 12

    new_profile = recommendations_engine.get_tax_profile(modified_user)

    st.markdown("### Opportunity Analysis")

    o1, o2, o3 = st.columns(3)
    with o1:
        st.metric("Total Est. Tax Savings Opportunity", f"${new_profile['estimated_tax_savings']:,.0f}")
    with o2:
        st.metric("Opportunity Level", new_profile['opportunity_label'])
    with o3:
        st.metric("Preferred Strategy", new_profile['preferred_strategy'])

    if new_profile['actions']:
        st.markdown("#### Recommended Next Actions")
        for action in new_profile['actions']:
            st.success(f"**{action['account']}**: ${action['remaining_room']:,.0f} room left → Est. ${action['estimated_tax_savings']:,.0f} tax savings this year.")
    else:
        st.info("You're already maxing out the major tax-advantaged accounts. Great job!")

    st.caption("Calculations are estimates based on 2025/2026 US tax brackets and contribution limits. Consult a tax professional for personalized advice.")
