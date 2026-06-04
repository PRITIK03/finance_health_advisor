# AI-Powered Personal Finance Health Advisor

## Project Overview

This is an end-to-end machine learning data analysis project that provides comprehensive financial health analysis using synthetic data. The system generates realistic financial data for multiple users and applies various ML techniques to:

1. **Segment users** into financial health categories (clustering)
2. **Predict spending behavior** patterns (regression/forecasting)
3. **Classify financial risk levels** (classification)
4. **Detect anomalies** in spending patterns (anomaly detection)
5. **Generate personalized recommendations** (rule-based + ML)

## Unique Features

- **Multi-dimensional Analysis**: Combines clustering, classification, regression, and anomaly detection
- **Synthetic Data Generation**: Creates realistic financial data with patterns, trends, and anomalies
- **Interactive Dashboard**: Visual analytics with Streamlit web interface (27 pages)
- **Time-series Analysis**: Monthly spending trends and forecasting
- **Risk Assessment**: Financial health scoring system with 5 risk categories
- **Strategic Planning Tools**: What-If Simulator, Tax Optimizer, Monte Carlo Simulator
- **Advanced Analytics**: Peer benchmarking, anomaly detection, subscription audit

## Technology Stack

- **Python 3.10+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: ML algorithms (K-Means, Random Forest, Gradient Boosting, Isolation Forest, KNN)
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive charts
- **Streamlit**: Web UI dashboard
- **Faker**: Synthetic data generation

## Project Structure

```
finance_health_advisor/
├── app.py                    # Streamlit app shell (data loading, sidebar, routing)
├── main.py                   # CLI entrypoint for the full ML pipeline
├── config.py                 # Centralized configuration (env-overridable)
├── components.py             # Reusable UI widgets (metrics, cards, tabs, sparklines)
├── styles.py                 # Centralized theme CSS (light/dark mode)
├── data_generator.py         # Synthetic financial data (10,000 users)
├── preprocessing.py          # Feature engineering + simulation backends
├── models.py                 # ML pipelines (clustering, classification, etc.)
├── visualizations.py         # Plotly chart library
├── recommendations.py        # Rule-based + AI recommendation engine
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── SPEC.md                   # This file
├── __init__.py
├── __main__.py
├── .env.example              # Environment variable template
└── pages/                    # Modular UI pages (27)
    ├── __init__.py           # Page exports
    ├── router.py             # PAGE_ROUTES mapping (page name -> renderer)
    ├── dashboard.py          # Dashboard overview
    ├── stress_test.py        # Financial stress simulator
    ├── wealth_projection.py  # Compound interest projection
    ├── comparison.py         # Side-by-side user comparison
    ├── user_segmentation.py  # K-Means cluster profiles
    ├── risk_prediction.py    # Random Forest risk analysis
    ├── forecasting_page.py   # Savings forecasting
    ├── anomaly_detection.py  # Isolation Forest anomalies
    ├── recommendations_page.py # AI recommendations
    ├── goal_planner.py       # Financial goal roadmap
    ├── fire_tracker.py       # FIRE calculator
    ├── debt_optimizer.py     # Avalanche vs Snowball
    ├── data_viz.py           # Distribution / correlation / trend charts
    ├── financial_tips.py     # Educational expanders
    ├── document_vault.py     # Document upload demo
    ├── expense_classification.py # Spending breakdown
    ├── predictive_analytics.py  # Forecast + Monte Carlo suite
    ├── alerts.py             # Threshold-based alerts
    ├── peer_benchmarking.py  # KNN peer comparison
    ├── scenario_simulator.py # What-if Monte Carlo placeholder
    ├── data_explorer.py      # Raw data + CSV export
    ├── what_if_simulator.py  # Live budget slider simulator
    ├── tax_optimizer.py      # 401(k)/HSA/IRA planner
    ├── monte_carlo_simulator.py # Stochastic projections
    ├── subscription_manager.py  # Subscription audit
    ├── net_worth_tracker.py  # Net worth trend tracker
    └── charts.py             # Shared chart utilities
```

## Data Schema

### User Financial Data (10,000 users)
- user_id, age, income, employment_type
- monthly_expenses, savings, investments
- spending_categories (10 categories)
- credit_score, debt, loan_payments
- subscription_count, lifestyle_score
- financial_health_score, risk_label, cluster

### Monthly Transaction Data (12 months per user)
- month, income_variation, expense_variation
- spending_by_category, savings_rate
- financial_health_score, risk_label
- is_anomaly flag, ml_anomaly flag
- Derived features: quarter, spending_pct, concentration, rolling averages

## ML Models

1. **K-Means Clustering**: Segment users into 5 financial health tiers (Silhouette ~0.16)
2. **Random Forest Classifier**: Predict financial risk category (Accuracy ~94.5%)
3. **Gradient Boosting Regressor**: Forecast monthly savings (R² ~0.997)
4. **Isolation Forest**: Detect anomalous spending patterns (2% contamination)
5. **K-Nearest Neighbors**: Find similar financial profiles

## Configuration

All configurable parameters are centralized in `config.py` and can be overridden via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| FHA_TOTAL_USERS | Number of synthetic users | 10000 |
| FHA_ANOMALY_RATE | Anomaly injection rate | 0.015 |
| FHA_SEED | Random seed | 42 |
| FHA_N_CLUSTERS | K-Means clusters | 5 |
| FHA_RF_ESTIMATORS | Random Forest trees | 100 |
| FHA_IF_CONTAMINATION | Isolation Forest contamination | 0.02 |
| FHA_DARK_MODE | Default dark mode | false |
| FHA_PAGE_SIZE | Default page size | 100 |
| FHA_DEBUG | Enable debug mode | false |
| FHA_LOG_LEVEL | Logging level | INFO |

## Output Metrics

- Financial Health Score (0-100)
- Risk Category (Very Low, Low, Medium, High, Very High)
- Cluster Assignment (5 segments)
- Spending Pattern Analysis
- Personalized Recommendations
- Anomaly Alerts
- Monthly Forecasts
- FIRE projections
- Debt payoff comparisons
- Monte Carlo wealth distributions

## Getting Started

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

   Or run the CLI pipeline:
   ```bash
   python -m finance_health_advisor
   # or
   python main.py
   ```

## Quick Examples of Key Tools

Once the app is running, try these powerful modules from the sidebar:

- **🛠️ What-If Simulator**  
  Select a user → Drag the spending category sliders (e.g., reduce Subscriptions by 30%) → Watch your Health Score and projected savings update live.

- **💼 Tax Optimizer**  
  Pick a profile → Adjust your planned 401(k)/HSA/IRA contributions → Instantly see your estimated tax savings opportunity and remaining contribution room.

- **📉 Monte Carlo Simulator**  
  Enter starting capital and monthly contribution → Set expected return + volatility → Run 1000+ simulations → View the full distribution of possible outcomes (including worst-case 10th percentile).

These tools work entirely offline with the built-in synthetic dataset.

## UI Highlights

- **Dark Mode Support**: Seamlessly toggle between light and dark themes.
- **Interactive Gauges**: Professional KPI visualization for scores and progress.
- **Actionable Roadmap**: Step-by-step success paths for financial goals.
- **Clean Aesthetic**: Focused on readability, high-contrast text, and modern card layouts.

## Frontend Enhancements

- **🎨 Animated Gradient Header**: Dynamic purple/blue gradient with CSS animation
- **🖱️ Card Hover Effects**: Smooth 2px lift with shadow on hover transitions
- **🔘 Button Animations**: Scale transform on hover with box shadow feedback
- **📊 Styled Metric Cards**: Gradient backgrounds with left accent borders
- **🎯 Custom Progress Bars**: Gradient fills with rounded corners
- **📱 Responsive Design**: Mobile-friendly layout with adaptive containers

## Feature Overview

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
| 🔥 FIRE Tracker | Financial independence calculator | ✅ Active |
| 💸 Debt Optimizer | Avalanche vs Snowball comparison | ✅ Active |
| 🛠️ What-If Simulator | Live budget adjustments with real-time health score impact | ✅ Active |
| 💼 Tax Optimizer | Interactive tax-advantaged account planner & savings estimator | ✅ Active |
| 📉 Monte Carlo Simulator | Stochastic wealth projections with volatility and outcome distributions | ✅ Active |
| 📱 Subscription Manager | Track and optimize recurring subscription costs with AI-powered insights | ✅ Active |
| 💰 Net Worth Tracker | Track net worth over time with detailed asset and liability breakdowns | ✅ Active |

---

*Built for professional financial intelligence and actionable strategy.*