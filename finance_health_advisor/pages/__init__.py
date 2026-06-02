"""
Pages package for Finance Health Advisor
Each file contains render functions for a specific section.
"""
from .charts import (
    create_pie_chart,
    create_multi_line_chart,
    create_multi_bar_chart,
    create_projection_chart,
)
from .dashboard import render_dashboard
from .what_if_simulator import render_what_if_simulator
from .tax_optimizer import render_tax_optimizer
from .monte_carlo_simulator import render_monte_carlo_simulator
from .subscription_manager import render_subscription_manager
from .net_worth_tracker import render_net_worth_tracker
from .stress_test import render_stress_test
from .wealth_projection import render_wealth_projection
from .comparison import render_comparison
from .user_segmentation import render_user_segmentation
from .risk_prediction import render_risk_prediction
from .forecasting_page import render_forecasting
from .anomaly_detection import render_anomaly_detection
from .recommendations_page import render_recommendations
from .goal_planner import render_goal_planner
from .fire_tracker import render_fire_tracker
from .debt_optimizer import render_debt_optimizer
from .data_viz import render_data_visualization
from .financial_tips import render_financial_tips
from .document_vault import render_document_vault
from .expense_classification import render_expense_classification
from .predictive_analytics import render_predictive_analytics
from .alerts import render_alerts
from .peer_benchmarking import render_peer_benchmarking
from .scenario_simulator import render_scenario_simulator
from .data_explorer import render_data_explorer

__all__ = [
    "render_dashboard",
    "render_what_if_simulator",
    "render_tax_optimizer",
    "render_monte_carlo_simulator",
    "render_subscription_manager",
    "render_net_worth_tracker",
    "render_stress_test",
    "render_wealth_projection",
    "render_comparison",
    "render_user_segmentation",
    "render_risk_prediction",
    "render_forecasting",
    "render_anomaly_detection",
    "render_recommendations",
    "render_goal_planner",
    "render_fire_tracker",
    "render_debt_optimizer",
    "render_data_visualization",
    "render_financial_tips",
    "render_document_vault",
    "render_expense_classification",
    "render_predictive_analytics",
    "render_alerts",
    "render_peer_benchmarking",
    "render_scenario_simulator",
    "render_data_explorer",
    "create_pie_chart",
    "create_multi_line_chart",
    "create_multi_bar_chart",
    "create_projection_chart",
]
