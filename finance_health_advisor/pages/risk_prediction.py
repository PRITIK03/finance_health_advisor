"""
Risk Prediction Page Module
Financial Risk Analysis with Random Forest classifier results.
"""
import streamlit as st
import pandas as pd
from components import UIComponents
from preprocessing import prepare_classification_data
from sklearn.ensemble import RandomForestClassifier


def render_risk_prediction(users_df, monthly_df, visualizer, results):
    """Render the Financial Risk Analysis page."""
    UIComponents.page_header(
        "Financial Risk Analysis",
        "Random Forest classifier predicts financial vulnerability by analyzing user profiles and historical data.",
        icon="🎯"
    )
    UIComponents.info_box("Random Forest classifier predicts financial vulnerability by analyzing user profiles and historical data.")

    UIComponents.metric_row([
        {"label": "Training Accuracy", "value": f"{results['classification']['train_metrics']['accuracy']:.1%}"},
        {"label": "Test Accuracy", "value": f"{results['classification']['test_metrics']['accuracy']:.1%}"},
        {"label": "Model Reliability", "value": f"{results['classification']['test_metrics']['cv_mean']:.2f}"},
    ], columns=3)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 Prediction Insights", "📊 Importance & Distribution"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🎯 Risk Category Distribution</p>", unsafe_allow_html=True)
                risk_counts = users_df['risk_label'].value_counts()
                colors = {'Very Low': '#10b981', 'Low': '#34d399', 'Medium': '#f59e0b',
                         'High': '#f97316', 'Very High': '#ef4444'}

                import plotly.express as px
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    color=risk_counts.index,
                    color_discrete_map=colors,
                    hole=0.6
                )
                UIComponents.plotly_defaults(fig, height=350)
                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.markdown("<p class='card-title'>🧠 Risk Intelligence</p>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='sidebar-text'>
                The AI model classifies users into 5 risk levels by analyzing <b>{len(users_df.columns)}</b> financial dimensions.
                <br><br>
                <b>Top Risk Factors:</b>
                <ul>
                    <li><span class='highlight'>Credit Score</span>: Impact on borrowing capacity</li>
                    <li><span class='highlight'>Monthly Savings</span>: Buffer against volatility</li>
                    <li><span class='highlight'>Debt-to-Income</span>: Leverage sustainability</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        with st.container(border=True):
            st.markdown("<p class='card-title'>📊 Model Feature Importance</p>", unsafe_allow_html=True)
            X_class, y_class, le = prepare_classification_data(users_df)

            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_class.values, y_class)

            importance = pd.DataFrame({
                'feature': X_class.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=True)

            import plotly.express as px
            fig = px.bar(
                importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Blues'
            )
            UIComponents.plotly_defaults(fig, height=500)
            st.plotly_chart(fig, use_container_width=True)
