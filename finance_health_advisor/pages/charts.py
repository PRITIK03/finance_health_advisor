"""
Shared chart utilities for pages package
"""
import plotly.express as px
import plotly.graph_objects as go


def create_pie_chart(values, names, title=None, hole=0.6, color_sequence=None):
    """Create a pie chart with consistent styling."""
    if color_sequence is None:
        color_sequence = px.colors.qualitative.Set3

    fig = px.pie(
        values=values,
        names=names,
        hole=hole,
        color_discrete_sequence=color_sequence
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_multi_line_chart(x_data, y_data_dict, title, x_title, y_title, height=None):
    """Create a line chart with multiple series."""
    fig = go.Figure()

    for name, (y_values, line_props) in y_data_dict.items():
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_values,
            mode='lines+markers' if 'markers' in line_props else 'lines',
            name=name,
            line=line_props.get('line', dict(width=2)),
            fill=line_props.get('fill'),
            fillcolor=line_props.get('fillcolor')
        ))

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(t=30, b=30, l=30, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    if height:
        fig.update_layout(height=height)

    return fig


def create_multi_bar_chart(x_data, y_data_dict, title, x_title, y_title):
    """Create a bar chart with multiple series."""
    fig = go.Figure()

    for name, (y_values, marker_props) in y_data_dict.items():
        fig.add_trace(go.Bar(
            x=x_data,
            y=y_values,
            name=name,
            marker_color=marker_props.get('color')
        ))

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(t=30, b=30, l=30, r=30),
        barmode='relative',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_projection_chart(historical_months, historical_values, projection_months, projection_values,
                          goal_lines=None, title="Projection Chart"):
    """Create a projection chart with historical data, projection, and goal lines."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical_months,
        y=historical_values,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#10b981', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=projection_months,
        y=projection_values,
        mode='lines',
        name='Projection',
        line=dict(color='#3b82f6', width=3, dash='dash')
    ))

    if goal_lines:
        for goal_value, goal_dash, goal_color, goal_annotation, goal_position in goal_lines:
            fig.add_hline(
                y=goal_value,
                line_dash=goal_dash,
                line_color=goal_color,
                annotation_text=goal_annotation,
                annotation_position=goal_position
            )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Month",
        yaxis_title="Net Worth ($)",
        hovermode="x unified",
        margin=dict(t=30, b=30, l=30, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig