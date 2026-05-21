# 💰 AI-Powered Finance Health Advisor

A professional-grade financial advisory dashboard built with **Streamlit** and **Machine Learning**. This application provides a comprehensive "Wealth Intelligence" platform, offering actionable insights into financial health, risk prediction, and strategic planning.

**New in latest release**: What-If Budget Simulator, Tax Optimization Calculator, and full Monte Carlo stochastic projections.

## 🚀 Key Features

### 📊 Intelligence Dashboard
- **Financial Health Index**: Real-time health scoring using multivariate analysis.
- **Risk Prediction**: AI-powered classification of financial vulnerability (Random Forest).
- **Behavioral Analysis**: K-Means clustering to identify distinct financial profiles.
- **Micro-Charts**: High-density sparklines and mini-bullet charts for instant visual context.
- **💸 Subscription Audit**: Identify recurring cost leaks with usage-based AI analysis.
- **🏆 Gamification**: Achievement badges like "Debt-Free Dynamo" and "FIRE Starter".

### 💸 Strategic Planning Tools
- **🛠️ What-If Simulator** (NEW): Live budget adjustments with real-time impact on your Financial Health Score.
- **💼 Tax Optimizer** (NEW): Interactive planner to maximize 401(k), HSA, and IRA contributions with estimated tax savings.
- **📉 Monte Carlo Simulator** (NEW): Full stochastic projections with volatility — see 10th/50th/90th percentile outcomes.
- **🔥 FIRE Tracker**: Inflation-adjusted roadmaps and safe withdrawal simulations.
- **💸 Debt Optimizer**: Compare Avalanche vs Snowball payoff strategies.
- **🎯 Goal Planner**: Milestone tracking with projections.
- **🚀 Wealth Projection**: Compound interest calculator (use Monte Carlo for realistic ranges).

### 🔍 Advanced Analytics
- **👥 Comparison Mode**: Side-by-side user profiling and comparative radar analysis.
- **👥 Peer Benchmarking**: Compare against similar neighbors (KNN).
- **🚨 Anomaly Detection**: Flag unusual spending patterns.
- **💡 Recommendations**: Personalized advice including **Tax Optimization** strategies.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with **Glassmorphism UI** (CSS backdrop-blur).
- **Data Science**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/).
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Random Forest, Gradient Boosting, K-Means, KNN, Isolation Forest).
- **Visualizations**: [Plotly](https://plotly.com/) for interactive charts and gauges.
- **Styling**: Inter Font family, Dark Mode toggle, and responsive layout.

## 📂 Project Structure
 
- `app.py`: Main Streamlit orchestrator (lightweight entrypoint).
- `pages/`: Modular UI sections (dashboard.py and future page modules for better maintainability).
- `models.py`: Machine Learning pipelines and model training logic.
- `preprocessing.py`: Data engineering, financial ratios, and simulation backends.
- `visualizations.py`: Centralized module for all professional-grade charts and gauges.
- `recommendations.py`: Rule-based and AI-driven recommendation engine.
- `data_generator.py`: Synthetic data engine for large-scale financial simulation (10,000+ users).

**Note:** All external CDN dependencies for UI icons have been removed for full offline / air-gapped compatibility.

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

### Quick Examples of New Tools (v3.0)

Once the app is running, try these powerful new modules from the sidebar:

- **🛠️ What-If Simulator**  
  Select a user → Drag the spending category sliders (e.g., reduce Subscriptions by 30%) → Watch your Health Score and projected savings update live.

- **💼 Tax Optimizer**  
  Pick a profile → Adjust your planned 401(k)/HSA/IRA contributions → Instantly see your estimated tax savings opportunity and remaining contribution room.

- **📉 Monte Carlo Simulator**  
  Enter starting capital and monthly contribution → Set expected return + volatility → Run 1000+ simulations → View the full distribution of possible outcomes (including worst-case 10th percentile).

These tools work entirely offline with the built-in synthetic dataset.

## 🌙 UI Highlights
- **Dark Mode Support**: Seamlessly toggle between light and dark themes.
- **Interactive Gauges**: Professional KPI visualization for scores and progress.
- **Actionable Roadmap**: Step-by-step success paths for financial goals.
- **Clean Aesthetic**: Focused on readability, high-contrast text, and modern card layouts.

## ✨ Frontend Enhancements (v2.0)
- **🎨 Animated Gradient Header**: Dynamic purple/blue gradient with CSS animation
- **🖱️ Card Hover Effects**: Smooth 2px lift with shadow on hover transitions
- **🔘 Button Animations**: Scale transform on hover with box shadow feedback
- **📊 Styled Metric Cards**: Gradient backgrounds with left accent borders
- **🎯 Custom Progress Bars**: Gradient fills with rounded corners
- **📱 Responsive Design**: Mobile-friendly layout with adaptive containers

## 📸 Feature Overview

| Feature | Description | Status |
|---------|-------------|--------|
| 📊 Dashboard Overview | Financial executive summary with KPIs | ✅ Active |
| 🚨 Stress Test | Financial shock simulation | ✅ Active |
| 🚀 Wealth Projection | Compound interest calculator | ✅ Active |
| 👥 Comparison Mode | Side-by-side user analysis | ✅ Active |
| 👥 User Segmentation | K-Means clustering profiles | ✅ Active |
| 🎯 Risk Prediction | Random Forest classification | ✅ Active |
| 📈 Forecasting | Savings prediction & trends | ✅ Active |
| 🚨 Anomaly Detection | Isolation Forest outliers | ✅ Active |
| 💡 Recommendations | AI-powered financial advice | ✅ Active |
| 🎯 Goal Planner | Milestone tracking with projections | ✅ Active |
| 🔥 FIRE Tracker | Financial independence calculator | ✅ **NEW** |
| 💸 Debt Optimizer | Avalanche vs Snowball comparison | ✅ **NEW** |
| 🛠️ What-If Simulator | Live budget adjustments with real-time health score impact | ✅ **NEW** |
| 💼 Tax Optimizer | Interactive tax-advantaged account planner & savings estimator | ✅ **NEW** |
| 📉 Monte Carlo Simulator | Stochastic wealth projections with volatility and outcome distributions | ✅ **NEW** |

---
*Built for professional financial intelligence and actionable strategy.*
