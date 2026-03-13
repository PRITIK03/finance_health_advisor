"""
Synthetic Financial Data Generator
Generates realistic financial data for 10,000 users with various patterns and anomalies.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Spending categories with realistic weights
SPENDING_CATEGORIES = [
    'Housing', 'Transportation', 'Food', 'Healthcare', 
    'Entertainment', 'Shopping', 'Education', 'Subscriptions',
    'Insurance', 'Miscellaneous'
]

EMPLOYMENT_TYPES = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 'Retired']

RISK_LABELS = ['Very Low', 'Low', 'Medium', 'High', 'Very High']


def generate_user_profile(user_id: int) -> dict:
    """Generate a single user's financial profile."""
    
    # Age affects income and financial behavior
    age = np.random.normal(38, 12)
    age = max(18, min(70, int(age)))
    
    # Employment type influences income distribution
    employment_type = random.choices(
        EMPLOYMENT_TYPES, 
        weights=[0.55, 0.15, 0.18, 0.07, 0.05]
    )[0]
    
    # Base income by age and employment
    if employment_type == 'Full-time':
        base_income = np.random.lognormal(10.5, 0.4)  # ~40k-150k
    elif employment_type == 'Part-time':
        base_income = np.random.lognormal(9.5, 0.4)  # ~15k-60k
    elif employment_type == 'Self-employed':
        base_income = np.random.lognormal(10.8, 0.6)  # More variance
    elif employment_type == 'Retired':
        base_income = np.random.lognormal(9.8, 0.3)  # ~20k-50k (pension/retirement)
    else:
        base_income = np.random.lognormal(9.0, 0.3)  # Lower income
    
    # Monthly income with some variation
    monthly_income = base_income / 12
    
    # Expenses typically 60-80% of income
    expense_ratio = np.random.beta(4, 2) * 0.4 + 0.5  # 50-90%
    monthly_expenses = monthly_income * expense_ratio
    
    # Savings = income - expenses + some noise
    savings = monthly_income - monthly_expenses
    savings = max(0, savings + np.random.normal(0, savings * 0.1))
    
    # Investments based on savings capacity
    investment_capacity = savings * np.random.uniform(0.1, 0.5)
    if savings < 200:
        investments = 0
    else:
        investments = investment_capacity
    
    # Credit score influenced by savings and debt management
    base_credit = 650
    if savings > 1000:
        base_credit += random.randint(10, 50)
    elif savings < 200:
        base_credit -= random.randint(20, 60)
    credit_score = min(850, max(300, int(base_credit + np.random.normal(0, 30))))
    
    # Debt based on age and income
    if age < 30:
        debt = np.random.lognormal(9, 1)  # Student loans, car
    elif age < 45:
        debt = np.random.lognormal(10.5, 1)  # Mortgage, family
    else:
        debt = np.random.lognormal(9.5, 1)  # Less debt
    
    # Loan payments
    loan_ratio = np.random.uniform(0.05, 0.25)
    loan_payments = monthly_income * loan_ratio
    
    # Subscription count
    subscription_count = int(np.random.poisson(4))
    
    # Lifestyle score (0-100)
    lifestyle_score = int(np.random.normal(60, 15))
    lifestyle_score = max(0, min(100, lifestyle_score))
    
    return {
        'user_id': user_id,
        'age': age,
        'employment_type': employment_type,
        'monthly_income': round(monthly_income, 2),
        'monthly_expenses': round(monthly_expenses, 2),
        'monthly_savings': round(savings, 2),
        'monthly_investments': round(investments, 2),
        'credit_score': credit_score,
        'total_debt': round(debt, 2),
        'monthly_loan_payments': round(loan_payments, 2),
        'subscription_count': subscription_count,
        'lifestyle_score': lifestyle_score
    }


def generate_spending_breakdown(income: float, expenses: float) -> dict:
    """Generate spending breakdown across categories."""
    
    spending = {}
    remaining = expenses
    
    # Housing (largest expense)
    housing_ratio = np.random.uniform(0.25, 0.40)
    spending['Housing'] = remaining * housing_ratio
    remaining -= spending['Housing']
    
    # Food
    food_ratio = np.random.uniform(0.10, 0.20)
    spending['Food'] = remaining * food_ratio
    remaining -= spending['Food']
    
    # Transportation
    transport_ratio = np.random.uniform(0.08, 0.15)
    spending['Transportation'] = remaining * transport_ratio
    remaining -= spending['Transportation']
    
    # Healthcare
    health_ratio = np.random.uniform(0.03, 0.10)
    spending['Healthcare'] = remaining * health_ratio
    remaining -= spending['Healthcare']
    
    # Entertainment
    entertainment_ratio = np.random.uniform(0.05, 0.12)
    spending['Entertainment'] = remaining * entertainment_ratio
    remaining -= spending['Entertainment']
    
    # Shopping
    shopping_ratio = np.random.uniform(0.03, 0.10)
    spending['Shopping'] = remaining * shopping_ratio
    remaining -= spending['Shopping']
    
    # Education
    education_ratio = np.random.uniform(0.02, 0.08)
    spending['Education'] = remaining * education_ratio
    remaining -= spending['Education']
    
    # Subscriptions
    subscription_ratio = np.random.uniform(0.02, 0.05)
    spending['Subscriptions'] = remaining * subscription_ratio
    remaining -= spending['Subscriptions']
    
    # Insurance
    insurance_ratio = np.random.uniform(0.03, 0.08)
    spending['Insurance'] = remaining * insurance_ratio
    remaining -= spending['Insurance']
    
    # Miscellaneous (remaining)
    spending['Miscellaneous'] = max(0, remaining)
    
    return {k: round(v, 2) for k, v in spending.items()}


def calculate_financial_health_score(profile: dict) -> float:
    """Calculate financial health score (0-100)."""
    
    # Savings rate score (30% weight)
    savings_rate = profile['monthly_savings'] / profile['monthly_income'] if profile['monthly_income'] > 0 else 0
    savings_score = min(100, savings_rate * 200)
    
    # Expense ratio score (25% weight) - lower is better
    expense_ratio = profile['monthly_expenses'] / profile['monthly_income'] if profile['monthly_income'] > 0 else 1
    expense_score = max(0, 100 - (expense_ratio - 0.5) * 200)
    
    # Credit score (15% weight)
    credit_score = (profile['credit_score'] - 300) / 550 * 100
    
    # Investment ratio (15% weight)
    investment_ratio = profile['monthly_investments'] / profile['monthly_income'] if profile['monthly_income'] > 0 else 0
    investment_score = min(100, investment_ratio * 250)
    
    # Debt ratio (15% weight) - lower is better
    debt_ratio = profile['total_debt'] / (profile['monthly_income'] * 12) if profile['monthly_income'] > 0 else 0
    debt_score = max(0, 100 - debt_ratio * 50)
    
    # Weighted average
    health_score = (
        savings_score * 0.30 +
        expense_score * 0.25 +
        credit_score * 0.15 +
        investment_score * 0.15 +
        debt_score * 0.15
    )
    
    return round(max(0, min(100, health_score)), 2)


def get_risk_label(health_score: float) -> str:
    """Map health score to risk label."""
    if health_score >= 80:
        return 'Very Low'
    elif health_score >= 60:
        return 'Low'
    elif health_score >= 40:
        return 'Medium'
    elif health_score >= 20:
        return 'High'
    else:
        return 'Very High'


def generate_monthly_data(user_profile: dict, month_num: int) -> dict:
    """Generate monthly transaction data with realistic variations."""
    
    base_income = user_profile['monthly_income']
    base_expenses = user_profile['monthly_expenses']
    
    # Add seasonality and trends
    month_factor = 1 + 0.1 * np.sin(2 * np.pi * month_num / 12)  # Seasonal
    trend_factor = 1 + 0.005 * month_num  # Slight upward trend
    random_factor = np.random.normal(1, 0.08)
    
    income_variation = base_income * month_factor * trend_factor * random_factor
    
    # Expense variation with some correlation to income
    expense_random = np.random.normal(1, 0.05)
    expense_variation = base_expenses * expense_random
    
    # Calculate savings
    savings = income_variation - expense_variation
    
    # Generate spending breakdown
    spending_breakdown = generate_spending_breakdown(income_variation, expense_variation)
    
    # Savings rate
    savings_rate = savings / income_variation if income_variation > 0 else 0
    
    # Create monthly record
    monthly_data = {
        'user_id': user_profile['user_id'],
        'month': month_num,
        'income': round(income_variation, 2),
        'expenses': round(expense_variation, 2),
        'savings': round(savings, 2),
        'savings_rate': round(savings_rate, 4),
        **spending_breakdown
    }
    
    return monthly_data


def generate_anomalies(df: pd.DataFrame, anomaly_rate: float = 0.02) -> pd.DataFrame:
    """Inject anomalies into the dataset."""
    
    df = df.copy()
    anomaly_count = int(len(df) * anomaly_rate)
    
    # Random anomaly indices
    indices = np.random.choice(len(df), anomaly_count, replace=False)
    
    for idx in indices:
        anomaly_type = np.random.choice(['spike', 'drop', 'unusual_category'])
        
        if anomaly_type == 'spike':
            # Unusual high spending
            df.loc[idx, 'expenses'] *= np.random.uniform(1.5, 2.5)
            df.loc[idx, 'Entertainment'] *= np.random.uniform(2, 4)
        elif anomaly_type == 'drop':
            # Unusually low income
            df.loc[idx, 'income'] *= np.random.uniform(0.3, 0.6)
        else:
            # Unusual spending pattern
            df.loc[idx, 'Shopping'] *= np.random.uniform(3, 5)
    
    return df


def generate_full_dataset(num_users: int = 10000) -> tuple:
    """Generate complete dataset for all users and months."""
    
    print(f"Generating financial data for {num_users} users...")
    
    # Generate user profiles
    user_profiles = []
    for user_id in range(1, num_users + 1):
        profile = generate_user_profile(user_id)
        profile['financial_health_score'] = calculate_financial_health_score(profile)
        profile['risk_label'] = get_risk_label(profile['financial_health_score'])
        user_profiles.append(profile)
    
    users_df = pd.DataFrame(user_profiles)
    
    # Generate monthly data (12 months per user)
    monthly_records = []
    for _, user in users_df.iterrows():
        for month in range(1, 13):
            monthly_data = generate_monthly_data(user, month)
            
            # Recalculate health score for each month
            monthly_data['financial_health_score'] = calculate_financial_health_score({
                'monthly_income': monthly_data['income'],
                'monthly_expenses': monthly_data['expenses'],
                'monthly_savings': monthly_data['savings'],
                'monthly_investments': user['monthly_investments'],
                'credit_score': user['credit_score'],
                'total_debt': user['total_debt']
            })
            monthly_data['risk_label'] = get_risk_label(monthly_data['financial_health_score'])
            
            # Add anomaly flag for some months
            if np.random.random() < 0.02:
                monthly_data['is_anomaly'] = 1
            else:
                monthly_data['is_anomaly'] = 0
            
            monthly_records.append(monthly_data)
    
    monthly_df = pd.DataFrame(monthly_records)
    
    # Inject anomalies
    monthly_df = generate_anomalies(monthly_df, anomaly_rate=0.015)
    
    print(f"Generated {len(users_df)} user profiles")
    print(f"Generated {len(monthly_df)} monthly records")
    print(f"Financial Health Score range: {users_df['financial_health_score'].min():.1f} - {users_df['financial_health_score'].max():.1f}")
    print(f"Risk distribution:\n{users_df['risk_label'].value_counts()}")
    
    return users_df, monthly_df


if __name__ == "__main__":
    # Test data generation
    users_df, monthly_df = generate_full_dataset(1000)
    print("\nUser Data Sample:")
    print(users_df.head())
    print("\nMonthly Data Sample:")
    print(monthly_df.head())