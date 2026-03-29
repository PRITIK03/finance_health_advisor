# 💰 AI-Powered Finance Health Advisor

A professional-grade financial advisory dashboard built with **Streamlit** and **Machine Learning**. This application provides a comprehensive "Wealth Intelligence" platform, offering actionable insights into financial health, risk prediction, and strategic planning.

## 🚀 Key Features

### 📊 Intelligence Dashboard
- **Financial Health Index**: Real-time health scoring using multivariate analysis.
- **Risk Prediction**: AI-powered classification of financial vulnerability (Random Forest).
- **Behavioral Analysis**: K-Means clustering to identify distinct financial profiles.
- **Micro-Charts**: High-density sparklines and mini-bullet charts for instant visual context.

### 💸 Strategic Planning Tools
- **🔥 FIRE Tracker**: Calculate your Financial Independence number and roadmap using the 4% Rule.
- **💸 Debt Optimizer**: Compare **Avalanche** (interest-focused) vs. **Snowball** (balance-focused) strategies to eliminate debt.
- **🎯 Goal Planner**: Interactive roadmap for major milestones like homes, cars, or emergency funds.
- **🚀 Wealth Projection**: Monte Carlo simulations to project long-term portfolio growth under market volatility.

### 🔍 Advanced Analytics
- **🔮 Scenario Simulator**: "What-If" analysis to see how changes in income or debt affect your health score.
- **👥 Peer Benchmarking**: Compare your financial standing against similar users using K-Nearest Neighbors (KNN).
- **🚨 Anomaly Detection**: Isolation Forest algorithm to flag unusual spending or savings patterns.
- **💡 Recommendations**: Personalized budget, debt, and investment advice based on behavioral ML insights.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with custom CSS injection for modern UI/UX.
- **Data Science**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/).
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Random Forest, Gradient Boosting, K-Means, KNN, Isolation Forest).
- **Visualizations**: [Plotly](https://plotly.com/) for interactive charts and gauges.
- **Styling**: Inter Font family, Dark Mode toggle, and responsive layout.

## 📂 Project Structure

- `app.py`: Main Streamlit orchestrator and UI implementation.
- `models.py`: Machine Learning pipelines and model training logic.
- `preprocessing.py`: Data engineering, financial ratios, and simulation backends.
- `visualizations.py`: Centralized module for all professional-grade charts and gauges.
- `recommendations.py`: Rule-based and AI-driven recommendation engine.
- `data_generator.py`: Synthetic data engine for large-scale financial simulation (10,000+ users).

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd finance_health_advisor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🌙 UI Highlights
- **Dark Mode Support**: Seamlessly toggle between light and dark themes.
- **Interactive Gauges**: Professional KPI visualization for scores and progress.
- **Actionable Roadmap**: Step-by-step success paths for financial goals.
- **Clean Aesthetic**: Focused on readability, high-contrast text, and modern card layouts.

---
*Built for professional financial intelligence and actionable strategy.*
