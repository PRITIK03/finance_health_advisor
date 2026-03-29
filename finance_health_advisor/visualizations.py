"""
Visualization Module
Creates comprehensive charts and plots for financial data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly import express as px


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Modern Plotly Template
MODERN_TEMPLATE = "plotly_white"
CHART_COLORS = px.colors.qualitative.Prism
RISK_COLORS = {
    'Very Low': '#2ecc71',   # Emerald
    'Low': '#27ae60',        # Nephritis
    'Medium': '#f1c40f',     # Sunflower
    'High': '#e67e22',       # Carrot
    'Very High': '#e74c3c'   # Alizarin
}

# Common layout updates for consistency
def update_chart_layout(fig, title):
    fig.update_layout(
        template=MODERN_TEMPLATE,
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=20, r=20),
        font=dict(family="Inter, sans-serif")
    )


class FinancialVisualizer:
    """Comprehensive visualization class for financial data."""
    
    def __init__(self, users_df: pd.DataFrame, monthly_df: pd.DataFrame):
        self.users_df = users_df
        self.monthly_df = monthly_df
        
    def create_distribution_plots(self) -> dict:
        """Create distribution plots for key metrics."""
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Financial Health Score Distribution
        axes[0, 0].hist(self.users_df['financial_health_score'], bins=30, 
                        edgecolor='black', alpha=0.7, color='#3498db')
        axes[0, 0].set_title('Financial Health Score Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Health Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Monthly Income Distribution
        axes[0, 1].hist(self.users_df['monthly_income'], bins=30, 
                        edgecolor='black', alpha=0.7, color='#2ecc71')
        axes[0, 1].set_title('Monthly Income Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Income ($)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Savings Distribution
        axes[0, 2].hist(self.users_df['monthly_savings'], bins=30, 
                        edgecolor='black', alpha=0.7, color='#9b59b6')
        axes[0, 2].set_title('Monthly Savings Distribution', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Savings ($)')
        axes[0, 2].set_ylabel('Frequency')
        
        # Credit Score Distribution
        axes[1, 0].hist(self.users_df['credit_score'], bins=30, 
                        edgecolor='black', alpha=0.7, color='#e74c3c')
        axes[1, 0].set_title('Credit Score Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Credit Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # Expense Ratio Distribution
        axes[1, 1].hist(self.users_df['expense_ratio'], bins=30, 
                        edgecolor='black', alpha=0.7, color='#f39c12')
        axes[1, 1].set_title('Expense Ratio Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Expense Ratio')
        axes[1, 1].set_ylabel('Frequency')
        
        # Age Distribution
        axes[1, 2].hist(self.users_df['age'], bins=30, 
                        edgecolor='black', alpha=0.7, color='#1abc9c')
        axes[1, 2].set_title('Age Distribution', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Age')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        return {'distribution_plots': fig}
    
    def create_risk_analysis_plot(self) -> go.Figure:
        """Create risk category analysis visualization."""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Risk Category Distribution', 'Financial Health by Risk Category'),
            specs=[[{'type': 'pie'}, {'type': 'box'}]]
        )
        
        # Pie chart for risk distribution
        risk_counts = self.users_df['risk_label'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker_colors=[RISK_COLORS.get(r, '#95a5a6') for r in risk_counts.index],
                textinfo='percent+label',
                hole=0.4,
                pull=[0.05] * len(risk_counts),
                hovertemplate="<b>%{label} Risk</b><br>Users: %{value}<br>Percentage: %{percent}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Box plot for health score by risk
        for risk in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
            if risk in self.users_df['risk_label'].unique():
                risk_data = self.users_df[self.users_df['risk_label'] == risk]['financial_health_score']
                fig.add_trace(
                    go.Box(
                        y=risk_data,
                        name=risk,
                        marker_color=RISK_COLORS.get(risk, '#95a5a6'),
                        boxmean=True,
                        jitter=0.3,
                        pointpos=-1.8
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            showlegend=False,
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap for numeric features."""
        
        numeric_cols = ['monthly_income', 'monthly_expenses', 'monthly_savings',
                       'monthly_investments', 'credit_score', 'total_debt',
                       'monthly_loan_payments', 'financial_health_score']
        
        corr_matrix = self.users_df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Blues',
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='Feature Correlation Heatmap',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    def create_spending_breakdown_chart(self) -> go.Figure:
        """Create spending breakdown by category."""
        
        spending_cols = ['Housing', 'Transportation', 'Food', 'Healthcare', 
                        'Entertainment', 'Shopping', 'Education', 'Subscriptions',
                        'Insurance', 'Miscellaneous']
        
        # Average spending by category
        avg_spending = self.monthly_df[spending_cols].mean().sort_values(ascending=False)
        
        # Donut chart
        fig = go.Figure(data=[go.Pie(
            labels=avg_spending.index,
            values=avg_spending.values,
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=CHART_COLORS),
            pull=[0.02] * len(avg_spending),
            hovertemplate="<b>%{label}</b><br>Avg Spend: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>"
        )])
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='Average Spending Breakdown',
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    def create_time_series_plot(self) -> go.Figure:
        """Create monthly trends over time."""
        
        monthly_avg = self.monthly_df.groupby('month').agg({
            'income': 'mean',
            'expenses': 'mean',
            'savings': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_avg['month'],
            y=monthly_avg['income'],
            mode='lines+markers',
            name='Income',
            line=dict(color='#2ecc71', width=3, shape='spline')
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_avg['month'],
            y=monthly_avg['expenses'],
            mode='lines+markers',
            name='Expenses',
            line=dict(color='#e74c3c', width=3, shape='spline')
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_avg['month'],
            y=monthly_avg['savings'],
            mode='lines+markers',
            name='Savings',
            line=dict(color='#3498db', width=3, shape='spline')
        ))
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='Monthly Financial Trends (Average)',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_cluster_analysis_plot(self) -> go.Figure:
        """Create cluster analysis visualization."""
        
        if 'cluster' not in self.users_df.columns:
            return None
            
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cluster Distribution', 'Financial Metrics by Cluster')
        )
        
        # Cluster distribution
        cluster_counts = self.users_df['cluster'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                marker_color=CHART_COLORS,
                name='Users',
                text=cluster_counts.values,
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # Cluster metrics
        cluster_metrics = self.users_df.groupby('cluster').agg({
            'monthly_income': 'mean',
            'monthly_savings': 'mean',
            'financial_health_score': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=cluster_metrics['cluster'],
                y=cluster_metrics['monthly_income'],
                name='Avg Income',
                marker_color=CHART_COLORS[0]
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=cluster_metrics['cluster'],
                y=cluster_metrics['monthly_savings'],
                name='Avg Savings',
                marker_color=CHART_COLORS[1]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='Cluster Analysis',
            height=450,
            showlegend=True,
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    def create_anomaly_analysis_plot(self) -> go.Figure:
        """Create anomaly analysis visualization."""
        
        if 'ml_anomaly' not in self.monthly_df.columns:
            return None
            
        anomalies = self.monthly_df[self.monthly_df['ml_anomaly'] == 1]
        normal = self.monthly_df[self.monthly_df['ml_anomaly'] == 0]
        
        fig = go.Figure()
        
        # Normal transactions
        fig.add_trace(go.Scatter(
            x=normal['expenses'],
            y=normal['savings'],
            mode='markers',
            name='Normal',
            marker=dict(color='#3498db', size=5, opacity=0.4)
        ))
        
        # Anomalies
        fig.add_trace(go.Scatter(
            x=anomalies['expenses'],
            y=anomalies['savings'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#e74c3c', size=10, symbol='x', line=dict(width=1))
        ))
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='Anomaly Detection: Expenses vs Savings',
            xaxis_title='Expenses ($)',
            yaxis_title='Savings ($)',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_income_expense_scatter(self) -> go.Figure:
        """Create income vs expenses scatter plot with health score."""
        
        fig = go.Figure()
        
        # Color by health score
        fig.add_trace(go.Scatter(
            x=self.users_df['monthly_income'],
            y=self.users_df['monthly_expenses'],
            mode='markers',
            marker=dict(
                size=10,
                color=self.users_df['financial_health_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Health Score', thickness=15),
                line=dict(width=0.5, color='white')
            ),
            text=self.users_df.apply(
                lambda r: f"Income: ${r['monthly_income']:,.0f}<br>Expenses: ${r['monthly_expenses']:,.0f}<br>Score: {r['financial_health_score']:.1f}", 
                axis=1
            ),
            hoverinfo='text'
        ))
        
        # Add trend line
        z = np.polyfit(self.users_df['monthly_income'], self.users_df['monthly_expenses'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.users_df['monthly_income'].min(), 
                            self.users_df['monthly_income'].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            name='Trend Line',
            line=dict(color='rgba(0,0,0,0.5)', dash='dash', width=2)
        ))
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='Income vs Expenses',
            xaxis_title='Monthly Income ($)',
            yaxis_title='Monthly Expenses ($)',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif"),
            showlegend=False
        )
        
        return fig
    
    def create_employment_analysis(self) -> go.Figure:
        """Create analysis by employment type."""
        
        employment_stats = self.users_df.groupby('employment_type').agg({
            'financial_health_score': 'mean',
            'monthly_income': 'mean',
            'monthly_savings': 'mean'
        }).reset_index().sort_values('financial_health_score', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(name='Avg Income', x=employment_stats['employment_type'], 
                   y=employment_stats['monthly_income'], marker_color=CHART_COLORS[0]),
            go.Bar(name='Avg Savings', x=employment_stats['employment_type'], 
                   y=employment_stats['monthly_savings'], marker_color=CHART_COLORS[1]),
            go.Bar(name='Health Score (x100)', x=employment_stats['employment_type'], 
                   y=employment_stats['financial_health_score']*100, marker_color=CHART_COLORS[2])
        ])
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            barmode='group',
            title='Financial Metrics by Employment Type',
            xaxis_title='Employment Type',
            yaxis_title='Value',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig

    def create_peer_comparison_radar(self, user_data: pd.Series, peers_avg: pd.Series) -> go.Figure:
        """Create a radar chart comparing a user to their peers."""
        
        categories = ['Income', 'Expenses', 'Savings', 'Investments', 'Credit Score', 'Health Score']
        
        # Normalize values for radar chart (roughly 0-100 scale)
        def normalize(val, max_val):
            return min(100, (val / max_val * 100)) if max_val > 0 else 0

        # Estimate max values for normalization
        max_income = self.users_df['monthly_income'].quantile(0.95)
        max_expenses = self.users_df['monthly_expenses'].quantile(0.95)
        max_savings = self.users_df['monthly_savings'].quantile(0.95)
        max_investments = self.users_df['monthly_investments'].quantile(0.95)

        user_values = [
            normalize(user_data['monthly_income'], max_income),
            normalize(user_data['monthly_expenses'], max_expenses),
            normalize(user_data['monthly_savings'], max_savings),
            normalize(user_data['monthly_investments'], max_investments),
            (user_data['credit_score'] - 300) / 5.5, # 300-850 to 0-100
            user_data['financial_health_score']
        ]
        
        peer_values = [
            normalize(peers_avg['monthly_income'], max_income),
            normalize(peers_avg['monthly_expenses'], max_expenses),
            normalize(peers_avg['monthly_savings'], max_savings),
            normalize(peers_avg['monthly_investments'], max_investments),
            (peers_avg['credit_score'] - 300) / 5.5,
            peers_avg['financial_health_score']
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=user_values,
            theta=categories,
            fill='toself',
            name='You',
            line_color='#2563eb'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=peer_values,
            theta=categories,
            fill='toself',
            name='Peer Average',
            line_color='#94a3b8'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            title='You vs Peer Benchmark',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=80, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        return fig

    def create_debt_paydown_chart(self, results: dict) -> go.Figure:
        """Create debt paydown projection chart."""
        
        fig = go.Figure()
        
        avalanche_hist = results['avalanche']['history']
        snowball_hist = results['snowball']['history']
        
        max_months = max(len(avalanche_hist), len(snowball_hist))
        months = list(range(max_months))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=avalanche_hist,
            name='Debt Avalanche',
            line=dict(color='#3b82f6', width=3, shape='spline'),
            hovertemplate="Month %{x}<br>Balance: $%{y:,.0f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=snowball_hist,
            name='Debt Snowball',
            line=dict(color='#10b981', width=3, shape='spline', dash='dash'),
            hovertemplate="Month %{x}<br>Balance: $%{y:,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='Debt Elimination Projection',
            xaxis_title='Months to Debt-Free',
            yaxis_title='Remaining Balance ($)',
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig

    def create_interest_savings_chart(self, results: dict) -> go.Figure:
        """Create chart comparing total interest paid."""
        
        strategies = ['Avalanche', 'Snowball']
        interest = [results['avalanche']['total_interest'], results['snowball']['total_interest']]
        months = [results['avalanche']['months'], results['snowball']['months']]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Interest Paid", "Months to Pay Off"))
        
        fig.add_trace(
            go.Bar(x=strategies, y=interest, marker_color=['#3b82f6', '#10b981'], 
                   text=[f"${v:,.0f}" for v in interest], textposition='auto'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=strategies, y=months, marker_color=['#3b82f6', '#10b981'],
                   text=[f"{v} mos" for v in months], textposition='auto'),
            row=1, col=2
        )
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family="Inter, sans-serif")
        )
        
        return fig

    def create_wealth_projection_chart(self, simulation_results: np.ndarray, years: int) -> go.Figure:
        """Create a fan chart for Monte Carlo wealth projection."""
        
        # Calculate percentiles
        median = np.median(simulation_results, axis=0)
        p10 = np.percentile(simulation_results, 10, axis=0)
        p25 = np.percentile(simulation_results, 25, axis=0)
        p75 = np.percentile(simulation_results, 75, axis=0)
        p90 = np.percentile(simulation_results, 90, axis=0)
        
        x = list(range(years + 1))
        
        fig = go.Figure()
        
        # Add 10th-90th percentile area
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(p90) + list(p10)[::-1],
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='10th-90th Percentile',
            hoverinfo='skip'
        ))
        
        # Add 25th-75th percentile area
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(p75) + list(p25)[::-1],
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='25th-75th Percentile',
            hoverinfo='skip'
        ))
        
        # Add Median line
        fig.add_trace(go.Scatter(
            x=x,
            y=median,
            line=dict(color='#2563eb', width=3),
            name='Median Projection'
        ))
        
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='Monte Carlo Wealth Projection (Fan Chart)',
            xaxis_title='Years from Now',
            yaxis_title='Projected Wealth ($)',
            hovermode='x unified',
            showlegend=True,
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_gauge_chart(self, value: float, title: str, target: float = None) -> go.Figure:
        """Create a professional gauge chart for scores."""
        
        # Define color based on value
        if value >= 80:
            color = "#10b981" # Emerald
        elif value >= 60:
            color = "#34d399" # Green
        elif value >= 40:
            color = "#f59e0b" # Amber
        else:
            color = "#ef4444" # Red

        fig = go.Figure(go.Indicator(
            mode = "gauge+number" + ("+delta" if target else ""),
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 20, 'color': '#0f172a', 'weight': 'bold'}},
            delta = {'reference': target, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}} if target else None,
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#2563eb", 'width': 4},
                    'thickness': 0.75,
                    'value': target if target else value
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': "Inter, sans-serif"},
            margin=dict(t=50, b=20, l=30, r=30),
            height=300
        )

        return fig

    def create_fire_progress_chart(self, metrics: dict) -> go.Figure:
        """Create a chart showing progress towards FIRE goals."""
        
        categories = ['LeanFIRE', 'FIRE', 'FatFIRE']
        values = [metrics['lean_fire_number'], metrics['fire_number'], metrics['fat_fire_number']]
        current = metrics['current_investments']
        
        fig = go.Figure()
        
        # Add target bars
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            name='Target',
            marker_color='rgba(148, 163, 184, 0.2)',
            hovertemplate="Target: $%{y:,.0f}<extra></extra>"
        ))
        
        # Add current progress bar
        fig.add_trace(go.Bar(
            x=['Current Portfolio'],
            y=[current],
            name='Current',
            marker_color='#2563eb',
            hovertemplate="Current: $%{y:,.0f}<extra></extra>"
        ))
        
        # Add progress markers
        for i, val in enumerate(values):
            pct = (current / val) * 100 if val > 0 else 0
            fig.add_annotation(
                x=categories[i],
                y=val,
                text=f"{pct:.1f}%",
                showarrow=False,
                yshift=10,
                font=dict(color='#2563eb', weight='bold')
            )
            
        fig.update_layout(
            template=MODERN_TEMPLATE,
            title='FIRE Goal Comparison',
            yaxis_title='Amount ($)',
            barmode='overlay',
            showlegend=True,
            margin=dict(t=50, b=20, l=20, r=20),
            height=400
        )
        
        return fig
    
    def create_all_visualizations(self) -> dict:
        """Create all visualizations and return as dict."""
        
        return {
            'distributions': self.create_distribution_plots(),
            'risk_analysis': self.create_risk_analysis_plot(),
            'correlation': self.create_correlation_heatmap(),
            'spending_breakdown': self.create_spending_breakdown_chart(),
            'time_series': self.create_time_series_plot(),
            'cluster_analysis': self.create_cluster_analysis_plot(),
            'anomaly_analysis': self.create_anomaly_analysis_plot(),
            'income_expense': self.create_income_expense_scatter(),
            'employment': self.create_employment_analysis()
        }


def generate_summary_statistics(users_df: pd.DataFrame, monthly_df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset."""
    
    stats = {
        'total_users': len(users_df),
        'total_monthly_records': len(monthly_df),
        'avg_income': users_df['monthly_income'].mean(),
        'avg_expenses': users_df['monthly_expenses'].mean(),
        'avg_savings': users_df['monthly_savings'].mean(),
        'avg_health_score': users_df['financial_health_score'].mean(),
        'avg_credit_score': users_df['credit_score'].mean(),
        'risk_distribution': users_df['risk_label'].value_counts().to_dict()
    }
    
    return stats


if __name__ == "__main__":
    # Test visualizations
    from data_generator import generate_full_dataset
    
    users_df, monthly_df = generate_full_dataset(1000)
    visualizer = FinancialVisualizer(users_df, monthly_df)
    
    print("Creating visualizations...")
    viz = visualizer.create_all_visualizations()
    print(f"âœ“ Created {len(viz)} visualizations")
