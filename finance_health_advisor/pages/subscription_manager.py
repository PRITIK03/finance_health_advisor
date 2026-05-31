"""
Subscription Manager page module
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .charts import create_pie_chart, create_projection_chart


def render_subscription_manager(users_df, monthly_df, visualizer):
    """Render the Subscription Manager page."""
    st.header("📱 Subscription Manager")

    st.info("Track, analyze, and optimize your recurring subscription costs with AI-powered insights.")

    selected_user_id = st.selectbox(
        "Select User Profile",
        users_df['user_id'].unique(),
        format_func=lambda x: f"User {x}"
    )

    user_row = users_df[users_df['user_id'] == selected_user_id].iloc[0]
    user_monthly = monthly_df[monthly_df['user_id'] == selected_user_id]

    avg_sub_spend = float(user_monthly['Subscriptions'].mean()) if not user_monthly.empty else 50.0

    subscription_catalog = [
        {'name': 'Netflix Premium', 'category': 'Entertainment', 'cost': 22.99, 'frequency': 'Monthly', 'usage_score': 85, 'essential': False},
        {'name': 'Spotify Premium', 'category': 'Entertainment', 'cost': 10.99, 'frequency': 'Monthly', 'usage_score': 78, 'essential': False},
        {'name': 'Amazon Prime', 'category': 'Shopping', 'cost': 14.99, 'frequency': 'Monthly', 'usage_score': 92, 'essential': False},
        {'name': 'Gym Membership', 'category': 'Health & Fitness', 'cost': 45.00, 'frequency': 'Monthly', 'usage_score': 35, 'essential': False},
        {'name': 'YouTube Premium', 'category': 'Entertainment', 'cost': 11.99, 'frequency': 'Monthly', 'usage_score': 65, 'essential': False},
        {'name': 'Adobe Creative Cloud', 'category': 'Productivity', 'cost': 54.99, 'frequency': 'Monthly', 'usage_score': 72, 'essential': True},
        {'name': 'Microsoft 365', 'category': 'Productivity', 'cost': 12.99, 'frequency': 'Monthly', 'usage_score': 88, 'essential': True},
        {'name': 'Cloud Storage (1TB)', 'category': 'Productivity', 'cost': 9.99, 'frequency': 'Monthly', 'usage_score': 70, 'essential': False},
        {'name': 'New York Times', 'category': 'News', 'cost': 25.00, 'frequency': 'Monthly', 'usage_score': 45, 'essential': False},
        {'name': 'Meal Kit Delivery', 'category': 'Food', 'cost': 59.99, 'frequency': 'Weekly', 'usage_score': 40, 'essential': False},
        {'name': 'VPN Service', 'category': 'Security', 'cost': 12.99, 'frequency': 'Monthly', 'usage_score': 55, 'essential': False},
        {'name': 'Streaming Music Bundle', 'category': 'Entertainment', 'cost': 16.99, 'frequency': 'Monthly', 'usage_score': 82, 'essential': False},
        {'name': 'Gaming Subscription', 'category': 'Entertainment', 'cost': 14.99, 'frequency': 'Monthly', 'usage_score': 60, 'essential': False},
        {'name': 'Fitness App Premium', 'category': 'Health & Fitness', 'cost': 19.99, 'frequency': 'Monthly', 'usage_score': 50, 'essential': False},
        {'name': 'Cloud Backup', 'category': 'Security', 'cost': 6.99, 'frequency': 'Monthly', 'usage_score': 75, 'essential': True},
    ]

    selected_subs = []
    running_total = 0.0
    max_budget = max(avg_sub_spend * 1.5, 80.0)

    for sub in subscription_catalog:
        if running_total + sub['cost'] <= max_budget:
            selected_subs.append(sub.copy())
            running_total += sub['cost']

    if not selected_subs:
        selected_subs = subscription_catalog[:5]

    subs_df = pd.DataFrame(selected_subs)

    subs_df['annual_cost'] = subs_df.apply(
        lambda x: x['cost'] * 12 if x['frequency'] == 'Monthly' else x['cost'] * 52,
        axis=1
    )
    subs_df['usage_level'] = pd.cut(
        subs_df['usage_score'],
        bins=[0, 40, 70, 100],
        labels=['Low', 'Medium', 'High']
    )
    subs_df['recommendation'] = subs_df.apply(
        lambda x: 'Cancel' if x['usage_level'] == 'Low' and not x['essential']
        else ('Downgrade' if x['usage_level'] == 'Medium' and not x['essential']
              else 'Keep'),
        axis=1
    )
    subs_df['potential_monthly_savings'] = subs_df.apply(
        lambda x: x['cost'] if x['recommendation'] == 'Cancel'
        else (x['cost'] * 0.3 if x['recommendation'] == 'Downgrade' else 0),
        axis=1
    )

    total_monthly = subs_df['cost'].sum()
    total_annual = subs_df['annual_cost'].sum()
    potential_savings_monthly = subs_df['potential_monthly_savings'].sum()
    potential_savings_annual = potential_savings_monthly * 12
    low_usage_count = len(subs_df[subs_df['usage_level'] == 'Low'])
    essential_count = len(subs_df[subs_df['essential']])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Subscriptions", len(subs_df))
    with col2:
        st.metric("Monthly Cost", f"${total_monthly:,.2f}")
    with col3:
        st.metric("Potential Savings", f"${potential_savings_monthly:,.2f}/mo")
    with col4:
        st.metric("Low Usage Alerts", low_usage_count)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Subscription List", "💰 Cost Analysis", "🎯 Optimization"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>📊 Cost by Category</p>", unsafe_allow_html=True)

                category_costs = subs_df.groupby('category').agg({
                    'cost': 'sum',
                    'name': 'count'
                }).reset_index()
                category_costs.columns = ['Category', 'Monthly Cost', 'Count']
                category_costs = category_costs.sort_values('Monthly Cost', ascending=True)

                fig = px.bar(
                    category_costs,
                    x='Monthly Cost',
                    y='Category',
                    orientation='h',
                    color='Monthly Cost',
                    color_continuous_scale='Blues',
                    title='Monthly Spending by Category'
                )
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(t=30, b=30, l=30, r=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🎯 Usage Distribution</p>", unsafe_allow_html=True)

                usage_counts = subs_df['usage_level'].value_counts()
                colors = {'Low': '#ef4444', 'Medium': '#f59e0b', 'High': '#10b981'}

                fig = px.pie(
                    values=usage_counts.values,
                    names=usage_counts.index,
                    hole=0.6,
                    color=usage_counts.index,
                    color_discrete_map=colors
                )
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div style='text-align: center; color: #64748b;'>
                <b>{low_usage_count}</b> subscriptions with low usage detected
                </div>
                """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<p class='card-title'>💰 Annual Cost Breakdown</p>", unsafe_allow_html=True)

            fig = go.Figure(data=[
                go.Bar(
                    name='Monthly Cost',
                    x=subs_df['name'],
                    y=subs_df['cost'],
                    marker_color='#3b82f6'
                ),
                go.Bar(
                    name='Potential Savings',
                    x=subs_df['name'],
                    y=subs_df['potential_monthly_savings'],
                    marker_color='#10b981'
                )
            ])
            fig.update_layout(
                barmode='group',
                template="plotly_white",
                xaxis_title="Subscription",
                yaxis_title="Cost ($)",
                hovermode="x unified",
                margin=dict(t=30, b=30, l=30, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📋 Subscription Inventory</p>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                category_filter = st.multiselect(
                    "Filter by Category",
                    subs_df['category'].unique(),
                    default=subs_df['category'].unique()
                )
            with col2:
                usage_filter = st.multiselect(
                    "Filter by Usage",
                    subs_df['usage_level'].unique(),
                    default=subs_df['usage_level'].unique()
                )
            with col3:
                essential_filter = st.multiselect(
                    "Filter by Essential",
                    subs_df['essential'].unique(),
                    default=subs_df['essential'].unique()
                )

            filtered_df = subs_df[
                (subs_df['category'].isin(category_filter)) &
                (subs_df['usage_level'].isin(usage_filter)) &
                (subs_df['essential'].isin(essential_filter))
            ].copy()

            display_df = filtered_df[[
                'name', 'category', 'cost', 'frequency', 'annual_cost',
                'usage_score', 'usage_level', 'essential', 'recommendation',
                'potential_monthly_savings'
            ]].copy()

            display_df['essential'] = display_df['essential'].map({True: '✓', False: '✗'})
            display_df = display_df.rename(columns={
                'name': 'Name',
                'category': 'Category',
                'cost': 'Monthly Cost',
                'frequency': 'Frequency',
                'annual_cost': 'Annual Cost',
                'usage_score': 'Usage Score',
                'usage_level': 'Usage Level',
                'essential': 'Essential',
                'recommendation': 'Recommendation',
                'potential_monthly_savings': 'Potential Savings'
            })

            st.dataframe(display_df, use_container_width=True)

    with tab3:
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>💰 Cost Analysis Summary</p>", unsafe_allow_html=True)

                st.metric("Total Monthly Cost", f"${total_monthly:,.2f}")
                st.metric("Total Annual Cost", f"${total_annual:,.2f}")
                st.metric("Avg Cost per Subscription", f"${total_monthly/len(subs_df):,.2f}")
                st.metric("Essential Subscriptions", essential_count)

                st.markdown("---")
                st.write(f"**Percentage of Income:** {(total_monthly/user_row['monthly_income']*100):.1f}%")
                st.write(f"**Percentage of Expenses:** {(total_monthly/user_row['monthly_expenses']*100):.1f}%")

        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🎯 Savings Opportunities</p>", unsafe_allow_html=True)

                st.metric("Potential Monthly Savings", f"${potential_savings_monthly:,.2f}")
                st.metric("Potential Annual Savings", f"${potential_savings_annual:,.2f}")
                st.metric("Subscriptions to Cancel", len(subs_df[subs_df['recommendation'] == 'Cancel']))
                st.metric("Subscriptions to Downgrade", len(subs_df[subs_df['recommendation'] == 'Downgrade']))

                if potential_savings_annual > 500:
                    st.success(f"💡 You could save **${potential_savings_annual:,.0f}** annually by optimizing subscriptions!")
                elif potential_savings_annual > 200:
                    st.warning(f"💡 Moderate savings potential: **${potential_savings_annual:,.0f}** annually")
                else:
                    st.info("💡 Your subscription costs are well optimized!")

        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Cost vs Usage Scatter</p>", unsafe_allow_html=True)

            fig = px.scatter(
                subs_df,
                x='cost',
                y='usage_score',
                size='annual_cost',
                color='category',
                hover_data=['name', 'recommendation'],
                title='Cost vs Usage Analysis'
            )
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Monthly Cost ($)",
                yaxis_title="Usage Score (0-100)",
                margin=dict(t=30, b=30, l=30, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.info("🎯 **AI-Powered Optimization Recommendations**")

        with st.container(border=True):
            st.markdown("<p class='card-title'>🚨 Priority Actions</p>", unsafe_allow_html=True)

            cancel_subs = subs_df[subs_df['recommendation'] == 'Cancel'].sort_values('potential_monthly_savings', ascending=False)
            downgrade_subs = subs_df[subs_df['recommendation'] == 'Downgrade'].sort_values('potential_monthly_savings', ascending=False)

            if not cancel_subs.empty:
                st.subheader("Cancel These Subscriptions")
                for _, row in cancel_subs.iterrows():
                    st.error(f"**{row['name']}** (${row['cost']:.2f}/mo) - Low usage ({row['usage_score']}/100)")
                    st.caption(f"💡 Annual savings: ${row['annual_cost']:.2f}")

            if not downgrade_subs.empty:
                st.subheader("Consider Downgrading")
                for _, row in downgrade_subs.iterrows():
                    st.warning(f"**{row['name']}** (${row['cost']:.2f}/mo) - Medium usage ({row['usage_score']}/100)")
                    st.caption(f"💡 Potential savings: ${row['potential_monthly_savings']:.2f}/mo")

            if cancel_subs.empty and downgrade_subs.empty:
                st.success("🎉 No immediate action needed! Your subscriptions are well-optimized.")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<p class='card-title'>💡 Alternative Suggestions</p>", unsafe_allow_html=True)

            alternatives = [
                {
                    'current': 'Netflix Premium ($22.99)',
                    'alternative': 'Netflix Standard ($15.49)',
                    'savings': '$7.50/mo',
                    'note': 'If you don\'t need 4K streaming'
                },
                {
                    'current': 'Gym Membership ($45.00)',
                    'alternative': 'Home Workout Apps ($19.99)',
                    'savings': '$25.01/mo',
                    'note': 'If you prefer working out at home'
                },
                {
                    'current': 'Adobe Creative Cloud ($54.99)',
                    'alternative': 'Affinity Suite (One-time $169)',
                    'savings': '$459.88/year',
                    'note': 'If you don\'t need Adobe-specific features'
                },
                {
                    'current': 'Meal Kit Delivery ($59.99)',
                    'alternative': 'Meal Planning Apps ($9.99)',
                    'savings': '$50.00/mo',
                    'note': 'If you can cook yourself'
                }
            ]

            for alt in alternatives:
                st.markdown(f"""
                <div style='background-color: #f0f9ff; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #3b82f6;'>
                <b>Current:</b> {alt['current']}<br>
                <b>Alternative:</b> {alt['alternative']}<br>
                <b>Savings:</b> {alt['savings']}<br>
                <i>{alt['note']}</i>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<p class='card-title'>🎯 Set Subscription Budget</p>", unsafe_allow_html=True)

            budget_input = st.number_input(
                "Monthly Subscription Budget ($)",
                min_value=0,
                max_value=500,
                value=int(total_monthly),
                step=5
            )

            if total_monthly > budget_input:
                over_budget = total_monthly - budget_input
                st.warning(f"⚠️ You are **${over_budget:.2f}** over budget. Consider canceling {len(cancel_subs)} subscriptions.")
            else:
                under_budget = budget_input - total_monthly
                st.success(f"✅ You are **${under_budget:.2f}** under budget. Great job!")