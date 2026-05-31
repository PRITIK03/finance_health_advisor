"""
Pages package for Finance Health Advisor
Each file contains render functions for a specific section.
"""
from .subscription_manager import (
    _create_pie_chart,
    _create_multi_line_chart,
    _create_multi_bar_chart,
    _create_projection_chart,
)
from .dashboard import render_dashboard
from .what_if_simulator import render_what_if_simulator
from .tax_optimizer import render_tax_optimizer
from .monte_carlo_simulator import render_monte_carlo_simulator
from .subscription_manager import render_subscription_manager
from .net_worth_tracker import render_net_worth_tracker

__all__ = [
    "render_dashboard",
    "render_what_if_simulator",
    "render_tax_optimizer",
    "render_monte_carlo_simulator",
    "render_subscription_manager",
    "render_net_worth_tracker",
    "_create_pie_chart",
    "_create_multi_line_chart",
    "_create_multi_bar_chart",
    "_create_projection_chart",
]
