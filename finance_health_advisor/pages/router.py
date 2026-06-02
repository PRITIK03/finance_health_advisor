"""
Pages Router Module
Centralizes page routing for the Finance Health Advisor.
"""
from finance_health_advisor.pages.dashboard import render_dashboard
from finance_health_advisor.pages.stress_test import render_stress_test
from finance_health_advisor.pages.wealth_projection import render_wealth_projection
from finance_health_advisor.pages.comparison import render_comparison
from finance_health_advisor.pages.user_segmentation import render_user_segmentation
from finance_health_advisor.pages.risk_prediction import render_risk_prediction
from finance_health_advisor.pages.forecasting_page import render_forecasting
from finance_health_advisor.pages.anomaly_detection import render_anomaly_detection
from finance_health_advisor.pages.recommendations_page import render_recommendations
from finance_health_advisor.pages.goal_planner import render_goal_planner
from finance_health_advisor.pages.fire_tracker import render_fire_tracker
from finance_health_advisor.pages.debt_optimizer import render_debt_optimizer
from finance_health_advisor.pages.data_viz import render_data_visualization
from finance_health_advisor.pages.financial_tips import render_financial_tips
from finance_health_advisor.pages.document_vault import render_document_vault
from finance_health_advisor.pages.expense_classification import render_expense_classification
from finance_health_advisor.pages.predictive_analytics import render_predictive_analytics
from finance_health_advisor.pages.alerts import render_alerts
from finance_health_advisor.pages.peer_benchmarking import render_peer_benchmarking
from finance_health_advisor.pages.scenario_simulator import render_scenario_simulator
from finance_health_advisor.pages.data_explorer import render_data_explorer
from finance_health_advisor.pages.what_if_simulator import render_what_if_simulator
from finance_health_advisor.pages.tax_optimizer import render_tax_optimizer
from finance_health_advisor.pages.monte_carlo_simulator import render_monte_carlo_simulator
from finance_health_advisor.pages.subscription_manager import render_subscription_manager
from finance_health_advisor.pages.net_worth_tracker import render_net_worth_tracker


PAGE_ROUTES = {
    "📊 Dashboard Overview": lambda **kwargs: render_dashboard(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['recommendations_engine'], kwargs['visualizer']
    ),
    "🚨 Stress Test": lambda **kwargs: render_stress_test(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "🚀 Wealth Projection": lambda **kwargs: render_wealth_projection(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "👥 Comparison Mode": lambda **kwargs: render_comparison(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "👥 User Segmentation": lambda **kwargs: render_user_segmentation(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "🎯 Risk Prediction": lambda **kwargs: render_risk_prediction(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer'], kwargs['results']
    ),
    "📈 Forecasting": lambda **kwargs: render_forecasting(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer'], kwargs['results']
    ),
    "🚨 Anomaly Detection": lambda **kwargs: render_anomaly_detection(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "💡 Recommendations": lambda **kwargs: render_recommendations(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['recommendations_engine'], kwargs['visualizer']
    ),
    "🎯 Goal Planner": lambda **kwargs: render_goal_planner(
        kwargs['users_df'], kwargs['monthly_df']
    ),
    "🔥 FIRE Tracker": lambda **kwargs: render_fire_tracker(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "💸 Debt Optimizer": lambda **kwargs: render_debt_optimizer(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "📈 Data Visualization Dashboard": lambda **kwargs: render_data_visualization(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "📚 Financial Tips & Education": lambda **kwargs: render_financial_tips(),
    "🗂️ Document Upload & Storage": lambda **kwargs: render_document_vault(),
    "💸 Expense Categorization": lambda **kwargs: render_expense_classification(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "🔮 Predictive Analytics": lambda **kwargs: render_predictive_analytics(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer'], kwargs['results']
    ),
    "🔔 Alerts & Notifications": lambda **kwargs: render_alerts(
        kwargs['users_df'], kwargs['monthly_df']
    ),
    "👥 Peer Benchmarking": lambda **kwargs: render_peer_benchmarking(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "🔮 Scenario Simulator": lambda **kwargs: render_scenario_simulator(),
    "🔍 Data Explorer": lambda **kwargs: render_data_explorer(
        kwargs['users_df'], kwargs['monthly_df']
    ),
    "🛠️ What-If Simulator": lambda **kwargs: render_what_if_simulator(
        kwargs['users_df'], kwargs['monthly_df']
    ),
    "💼 Tax Optimizer": lambda **kwargs: render_tax_optimizer(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['recommendations_engine']
    ),
    "📉 Monte Carlo Simulator": lambda **kwargs: render_monte_carlo_simulator(
        kwargs['users_df'], kwargs['monthly_df']
    ),
    "📱 Subscription Manager": lambda **kwargs: render_subscription_manager(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
    "💰 Net Worth Tracker": lambda **kwargs: render_net_worth_tracker(
        kwargs['users_df'], kwargs['monthly_df'], kwargs['visualizer']
    ),
}
