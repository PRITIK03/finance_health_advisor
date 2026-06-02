"""
Reusable UI Components Module
Provides consistent, themed UI building blocks for the Finance Health Advisor.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any


class UIComponents:
    """Collection of reusable Streamlit UI components with consistent styling."""

    @staticmethod
    def page_header(title: str, subtitle: str, icon: str = "📊") -> None:
        """Render a consistent page header with icon, title, and subtitle."""
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(
                f"<div style='font-size: 3.5rem; text-align: center;'>{icon}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.title(title)
            st.markdown(
                f"<p style='font-size: 1.1rem; color: #64748b; margin-top: -15px;'>{subtitle}</p>",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    @staticmethod
    def section_card(title: str, icon: str = "📋", border: bool = True):
        """Create a styled card container for content sections."""
        card_title = f"{icon} {title}"
        return st.container(border=border if border else False), card_title

    @staticmethod
    def metric_row(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
        """Render a row of metric cards with consistent spacing."""
        cols = st.columns(columns)
        for idx, metric in enumerate(metrics):
            with cols[idx % columns]:
                st.metric(
                    metric.get("label", ""),
                    metric.get("value", ""),
                    delta=metric.get("delta", None),
                    help=metric.get("help", None),
                )

    @staticmethod
    def info_box(message: str, icon: str = "ℹ️") -> None:
        """Render an info box above page content."""
        st.info(f"{icon} {message}")

    @staticmethod
    def user_selector(users_df: pd.DataFrame, key: str = "user_select") -> pd.Series:
        """Render a user selection dropdown and return the selected user's data."""
        selected_user_id = st.selectbox(
            "Select User Profile",
            users_df["user_id"].unique(),
            format_func=lambda x: f"User {x}",
            key=key,
        )
        return users_df[users_df["user_id"] == selected_user_id].iloc[0]

    @staticmethod
    def insight_banner(message: str, category: str = "info") -> None:
        """Render a styled insight banner (success, warning, error, info)."""
        if category == "success":
            st.success(message)
        elif category == "warning":
            st.warning(message)
        elif category == "error":
            st.error(message)
        else:
            st.info(message)

    @staticmethod
    def recommendation_item(rec: Dict, expand: bool = False) -> None:
        """Render a single recommendation item with status icon."""
        status_icons = {
            "good": "🟢",
            "excellent": "💚",
            "warning": "🟡",
            "moderate": "🟠",
            "critical": "🔴",
            "info": "🔵",
        }
        icon = status_icons.get(rec.get("status", ""), "⚪")
        category = rec.get("category", "General")
        message = rec.get("message", "")
        suggestion = rec.get("suggestion", "")

        if expand:
            with st.expander(f"{icon} {category}", expanded=True):
                st.write(f"**{message}**")
                st.caption(f"💡 {suggestion}")
        else:
            st.markdown(f"**{icon} {category}**: {message}")
            st.markdown(f"   *💡 {suggestion}*")
            st.markdown("")

    @staticmethod
    def recommendations_tabs(
        recs: Dict[str, List[Dict]], tab_names: List[str] = None
    ) -> None:
        """Render recommendations grouped into tabs."""
        if tab_names is None:
            tab_names = ["💰 Budget", "💳 Debt", "🏦 Savings", "📈 Investments"]
        keys = ["budget", "debt", "savings", "investments"]
        tabs = st.tabs(tab_names)
        for idx, tab in enumerate(tabs):
            with tab:
                with st.container(border=True):
                    st.markdown(
                        f"<p class='card-title'>{tab_names[idx]}</p>",
                        unsafe_allow_html=True,
                    )
                    key = keys[idx] if idx < len(keys) else "general"
                    items = recs.get(key, [])
                    if not items:
                        st.caption("No recommendations available for this category.")
                    for rec in items:
                        UIComponents.recommendation_item(rec)

    @staticmethod
    def sparkline(
        data: list, color: str = "#2563eb", height: int = 40
    ) -> go.Figure:
        """Create a small sparkline chart for metric cards."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=data,
                mode="lines",
                fill="tozeroy",
                line=dict(color=color, width=2),
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
                hoverinfo="none",
            )
        )
        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        return fig

    @staticmethod
    def plotly_defaults(fig: go.Figure, height: int = 450) -> go.Figure:
        """Apply default Plotly styling to a figure."""
        fig.update_layout(
            template="plotly_white",
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif"),
        )
        return fig

    @staticmethod
    def gauge_chart(
        value: float, title: str, target: float = 90, height: int = 220
    ) -> go.Figure:
        """Create a themed gauge chart."""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": title, "font": {"size": 16, "weight": "bold"}},
                gauge={
                    "axis": {
                        "range": [None, 100],
                        "tickwidth": 1,
                        "tickcolor": "darkblue",
                    },
                    "bar": {"color": "#2563eb"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#e2e8f0",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(239, 68, 68, 0.1)"},
                        {"range": [40, 70], "color": "rgba(245, 158, 11, 0.1)"},
                        {"range": [70, 100], "color": "rgba(16, 185, 129, 0.1)"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": target,
                    },
                },
            )
        )
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )
        return fig

    @staticmethod
    def trend_metric_row(trends: Dict[str, tuple]) -> None:
        """Render a 2x2 grid of metric + sparkline pairs.

        trends: {label: (value, color, data_list)}
        """
        items = list(trends.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(items):
                    break
                label, val_tuple = items[idx]
                if len(val_tuple) == 3:
                    value, color, data = val_tuple
                else:
                    value, color = val_tuple
                    data = [value]
                with col:
                    st.metric(
                        label,
                        f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"
                        if isinstance(value, int)
                        else str(value),
                    )
                    if isinstance(data, (list, tuple)) and len(data) > 1:
                        fig = UIComponents.sparkline(data, color=color, height=35)
                        st.plotly_chart(
                            fig, use_container_width=True, config={"displayModeBar": False}
                        )
