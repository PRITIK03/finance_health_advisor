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
- **Interactive Dashboard**: Visual analytics with Streamlit web interface
- **Time-series Analysis**: Monthly spending trends and forecasting
- **Risk Assessment**: Financial health scoring system with 5 risk categories

## Technology Stack

- **Python 3.10+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: ML algorithms (K-Means, Random Forest, XGBoost, Isolation Forest)
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive charts
- **Streamlit**: Web UI dashboard
- **Faker**: Synthetic data generation

## Project Structure

```
finance-health-advisor/
├── main.py              # Main entry point
├── data_generator.py    # Synthetic data generation
├── preprocessing.py     # Data cleaning and feature engineering
├── models.py           # ML model implementations
├── visualizations.py   # Charts and plots
├── app.py              # Streamlit dashboard
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Data Schema

### User Financial Data (10,000 users)
- user_id, age, income, employment_type
- monthly_expenses, savings, investments
- spending_categories (10 categories)
- credit_score, debt, loan_payments
- subscription_count, lifestyle_score

### Monthly Transaction Data (12 months per user)
- month, income_variation, expense_variation
- spending_by_category, savings_rate
- financial_health_score, risk_label

## ML Models

1. **K-Means Clustering**: Segment users into 5 financial health tiers
2. **Random Forest Classifier**: Predict financial risk category
3. **XGBoost Regressor**: Forecast monthly savings
4. **Isolation Forest**: Detect anomalous spending patterns
5. **K-Nearest Neighbors**: Find similar financial profiles

## Output Metrics

- Financial Health Score (0-100)
- Risk Category (Very Low, Low, Medium, High, Very High)
- Spending Pattern Analysis
- Personalized Recommendations
- Anomaly Alerts
- Monthly Forecasts