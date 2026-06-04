"""
Data Preprocessing and Feature Engineering Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from config import CONFIG


class FinancialDataPreprocessor:
    """Preprocess and engineer features for financial data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.config = CONFIG
        
    def preprocess_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess user profile data."""
        
        df = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['employment_type', 'risk_label']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Feature engineering
        df = self._engineer_user_features(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        return df
    
    def _engineer_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for users."""
        
        # Financial ratios
        df['savings_rate'] = df['monthly_savings'] / df['monthly_income'].replace(0, 1)
        df['expense_ratio'] = df['monthly_expenses'] / df['monthly_income'].replace(0, 1)
        df['investment_rate'] = df['monthly_investments'] / df['monthly_income'].replace(0, 1)
        df['debt_to_income'] = df['total_debt'] / (df['monthly_income'] * 12).replace(0, 1)
        df['loan_to_income'] = df['monthly_loan_payments'] / df['monthly_income'].replace(0, 1)
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                  labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Income groups
        df['income_group'] = pd.qcut(df['monthly_income'], q=5, 
                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Disposable income
        df['disposable_income'] = df['monthly_income'] - df['monthly_expenses'] - df['monthly_loan_payments']
        
        # Financial stability score components
        df['stability_index'] = (
            (df['savings_rate'] > 0.1).astype(int) +
            (df['credit_score'] > 650).astype(int) +
            (df['investment_rate'] > 0.1).astype(int) +
            (df['debt_to_income'] < 0.3).astype(int)
        )
        
        return df
    
    def preprocess_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess monthly transaction data."""
        
        df = df.copy()
        
        # Add time-based features
        df['quarter'] = ((df['month'] - 1) // 3) + 1
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
        
        # Spending concentration
        spending_cols = ['Housing', 'Transportation', 'Food', 'Healthcare', 
                        'Entertainment', 'Shopping', 'Education', 'Subscriptions',
                        'Insurance', 'Miscellaneous']
        
        # Normalize spending to percentages
        df['total_spending'] = df[spending_cols].sum(axis=1)
        for col in spending_cols:
            df[col + '_pct'] = df[col] / df['total_spending'].replace(0, 1)
        
        # Spending concentration index (Herfindahl index)
        spending_pct_cols = [col + '_pct' for col in spending_cols]
        df['spending_concentration'] = (df[spending_pct_cols] ** 2).sum(axis=1)
        
        # Expense volatility (month-to-month variation)
        df = df.sort_values(['user_id', 'month'])
        df['expense_change'] = df.groupby('user_id')['expenses'].pct_change()
        df['income_change'] = df.groupby('user_id')['income'].pct_change()
        
        # Rolling averages (last 3 months)
        df['savings_ma3'] = df.groupby('user_id')['savings'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Scale numeric features."""
        
        return self.scaler.fit_transform(df[feature_cols])
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Calculate correlation-based feature importance."""
        
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
        
        return pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })


def calculate_fire_metrics(user_row: pd.Series, safe_withdrawal_rate: float = 0.04, inflation_rate: float = 0.03) -> dict:
    """Calculate Financial Independence, Retire Early (FIRE) metrics."""
    
    annual_expenses = user_row['monthly_expenses'] * 12
    current_investments = user_row['monthly_investments'] * 12 * 2 # Assumption: 2 years of investments already saved
    
    # FIRE Numbers
    fire_number = annual_expenses / safe_withdrawal_rate
    lean_fire_number = (annual_expenses * 0.7) / safe_withdrawal_rate
    fat_fire_number = (annual_expenses * 1.5) / safe_withdrawal_rate
    
    # Progress
    progress_pct = (current_investments / fire_number) * 100 if fire_number > 0 else 0
    
    # Real expected return (Nominal - Inflation)
    expected_return = 0.07 # 7% average market return
    real_return = expected_return - inflation_rate
    
    monthly_contribution = user_row['monthly_savings'] + user_row['monthly_investments']
    annual_contribution = monthly_contribution * 12
    
    years_to_fire = 0
    temp_wealth = current_investments
    if annual_contribution > 0:
        while temp_wealth < fire_number and years_to_fire < 100:
            temp_wealth = (temp_wealth + annual_contribution) * (1 + real_return)
            years_to_fire += 1
    else:
        years_to_fire = float('inf')

    return {
        'fire_number': fire_number,
        'lean_fire_number': lean_fire_number,
        'fat_fire_number': fat_fire_number,
        'current_investments': current_investments,
        'progress_pct': min(100, progress_pct),
        'years_to_fire': years_to_fire,
        'annual_expenses': annual_expenses
    }


def calculate_emergency_fund_stress_test(
    user_row: pd.Series,
    crisis_type: str = "Job Loss",
    emergency_fund_months: int = 6
) -> dict:
    """
    Estimate emergency fund resilience under common crises.
    crisis_type: 'Job Loss', 'Medical Emergency', 'Major Repair'
    """
    monthly_expenses = float(user_row['monthly_expenses'])
    initial_fund = max(0.0, float(user_row['monthly_savings']) * emergency_fund_months)

    crisis_profiles = {
        "Job Loss": {
            "expense_multiplier": 0.70,
            "one_time_cost": 0.0,
            "impact_message": "Discretionary spending is reduced by 30% during income disruption."
        },
        "Medical Emergency": {
            "expense_multiplier": 1.10,
            "one_time_cost": 5000.0,
            "impact_message": "A one-time medical bill hits first, then follow-up costs raise monthly spending."
        },
        "Major Repair": {
            "expense_multiplier": 1.00,
            "one_time_cost": 3000.0,
            "impact_message": "A large repair is paid upfront while monthly living costs stay unchanged."
        }
    }

    profile = crisis_profiles.get(crisis_type, crisis_profiles["Job Loss"])
    crisis_expenses = monthly_expenses * profile["expense_multiplier"]
    remaining_fund_start = max(0.0, initial_fund - profile["one_time_cost"])
    months_buffer = (
        remaining_fund_start / crisis_expenses if crisis_expenses > 0 else float('inf')
    )
    recommended_fund = crisis_expenses * 6
    fund_gap = max(0.0, recommended_fund - remaining_fund_start)

    balance_history = [remaining_fund_start]
    running_balance = remaining_fund_start
    for _ in range(1, 25):
        running_balance = max(0.0, running_balance - crisis_expenses)
        balance_history.append(running_balance)
        if running_balance <= 0:
            break

    if months_buffer >= 6:
        safety_band = "Strong"
    elif months_buffer >= 3:
        safety_band = "Moderate"
    else:
        safety_band = "Fragile"

    return {
        'months_buffer': months_buffer,
        'crisis_expenses': crisis_expenses,
        'initial_fund': initial_fund,
        'remaining_fund_start': remaining_fund_start,
        'impact_message': profile["impact_message"],
        'is_safe': months_buffer >= 6,
        'recommended_fund': recommended_fund,
        'fund_gap': fund_gap,
        'emergency_fund_months': emergency_fund_months,
        'one_time_cost': profile["one_time_cost"],
        'safety_band': safety_band,
        'balance_history': balance_history
    }


def calculate_subscription_audit(monthly_df: pd.DataFrame, user_id: int) -> tuple[pd.DataFrame, dict]:
    """Estimate recurring subscription leakage and optimization potential for a user."""
    user_monthly = monthly_df[monthly_df['user_id'] == user_id]
    avg_sub_budget = float(user_monthly['Subscriptions'].mean()) if not user_monthly.empty else 0.0

    catalog = [
        {'name': 'Netflix', 'cost': 19.99, 'usage_score': 88},
        {'name': 'Spotify', 'cost': 10.99, 'usage_score': 72},
        {'name': 'Gym Membership', 'cost': 45.00, 'usage_score': 28},
        {'name': 'Cloud Storage', 'cost': 9.99, 'usage_score': 81},
        {'name': 'News Subscription', 'cost': 15.00, 'usage_score': 42},
        {'name': 'Meal App Premium', 'cost': 12.99, 'usage_score': 35},
        {'name': 'Productivity Suite', 'cost': 14.99, 'usage_score': 67},
    ]

    selected = []
    running_total = 0.0
    threshold = max(avg_sub_budget * 1.35, 25.0)
    for item in catalog:
        if running_total + item['cost'] <= threshold:
            selected.append(item.copy())
            running_total += item['cost']

    if not selected:
        selected.append(catalog[0].copy())

    audit_df = pd.DataFrame(selected)
    audit_df['annual_cost'] = audit_df['cost'] * 12
    audit_df['usage'] = pd.cut(
        audit_df['usage_score'],
        bins=[-1, 39, 69, 100],
        labels=['Low', 'Medium', 'High']
    )
    audit_df['recommendation'] = audit_df['usage'].map({
        'Low': 'Cancel or downgrade',
        'Medium': 'Review and bundle',
        'High': 'Keep'
    })
    audit_df['potential_monthly_savings'] = np.where(
        audit_df['usage'] == 'Low',
        audit_df['cost'],
        np.where(audit_df['usage'] == 'Medium', audit_df['cost'] * 0.35, 0.0)
    )
    audit_df = audit_df.sort_values(['usage_score', 'cost'], ascending=[True, False]).reset_index(drop=True)

    summary = {
        'estimated_monthly_total': float(audit_df['cost'].sum()),
        'estimated_annual_total': float(audit_df['annual_cost'].sum()),
        'potential_monthly_savings': float(audit_df['potential_monthly_savings'].sum()),
        'potential_annual_savings': float(audit_df['potential_monthly_savings'].sum() * 12),
        'high_risk_count': int((audit_df['usage'] == 'Low').sum()),
        'services_count': int(len(audit_df))
    }

    return audit_df, summary


def calculate_debt_paydown(debts: list, extra_monthly_payment: float = 0) -> dict:
    """
    Simulate debt paydown using Snowball and Avalanche methods.
    debts: list of dicts {'name': str, 'balance': float, 'interest_rate': float, 'min_payment': float}
    """
    
    def simulate(debt_list, strategy='avalanche'):
        # Sort based on strategy
        if strategy == 'avalanche':
            # Highest interest rate first
            sorted_debts = sorted(debt_list, key=lambda x: x['interest_rate'], reverse=True)
        else:
            # Snowball: Smallest balance first
            sorted_debts = sorted(debt_list, key=lambda x: x['balance'])
            
        current_debts = [d.copy() for d in sorted_debts]
        total_balance_history = []
        months = 0
        total_interest_paid = 0
        
        while any(d['balance'] > 0 for d in current_debts) and months < 600: # 50 year cap
            months += 1
            available_extra = extra_monthly_payment
            monthly_total_balance = 0
            
            # 1. Apply interest and minimum payments
            for d in current_debts:
                if d['balance'] > 0:
                    interest = (d['balance'] * (d['interest_rate'] / 100)) / 12
                    d['balance'] += interest
                    total_interest_paid += interest
                    
                    # Pay minimum
                    payment = min(d['balance'], d['min_payment'])
                    d['balance'] -= payment
                
            # 2. Apply extra payment to the target debt
            for d in current_debts:
                if d['balance'] > 0 and available_extra > 0:
                    payment = min(d['balance'], available_extra)
                    d['balance'] -= payment
                    available_extra -= payment
            
            monthly_total_balance = sum(d['balance'] for d in current_debts)
            total_balance_history.append(monthly_total_balance)
            
        return {
            'months': months,
            'total_interest': total_interest_paid,
            'history': total_balance_history
        }

    avalanche_res = simulate(debts, 'avalanche')
    snowball_res = simulate(debts, 'snowball')
    
    return {
        'avalanche': avalanche_res,
        'snowball': snowball_res
    }


def calculate_financial_stress_test(
    user_row: pd.Series,
    scenario_name: str = "Job Loss (6 months)",
    simulation_months: int = 24
) -> dict:
    """
    Simulate various financial stress test scenarios for a user.
    Scenarios: 'Job Loss (6 months)', 'Medical Emergency', 'Market Downturn', 'Unexpected Large Expense'
    """
    monthly_income = float(user_row.get('monthly_income', 0))
    monthly_expenses = float(user_row.get('monthly_expenses', 0))
    monthly_savings = float(user_row.get('monthly_savings', 0))
    total_debt = float(user_row.get('total_debt', 0))
    current_investments = float(user_row.get('monthly_investments', 0)) * 12 * 2 # Assuming 2 years of investments
    emergency_fund = monthly_savings * 6 # A proxy for readily available cash

    # Ensure non-negative values for calculations
    monthly_income = max(0.0, monthly_income)
    monthly_expenses = max(0.0, monthly_expenses)
    monthly_savings = max(0.0, monthly_savings)
    total_debt = max(0.0, total_debt)
    current_investments = max(0.0, current_investments)
    emergency_fund = max(0.0, emergency_fund)

    # Initial state
    initial_liquid_assets = emergency_fund + monthly_savings # Simplified liquid assets
    initial_net_worth = initial_liquid_assets + current_investments - total_debt

    scenario_profiles = {
        "Job Loss (6 months)": {
            "income_reduction_pct": 1.0, # 100% income loss
            "expense_increase_pct": 0.0,
            "one_time_cost": 0.0,
            "investment_impact_pct": 0.0,
            "duration_months": 6, # Duration of income loss
            "message": "Simulates 6 months of complete income loss, relying on savings and reduced spending."
        },
        "Medical Emergency": {
            "income_reduction_pct": 0.0,
            "expense_increase_pct": 0.10, # 10% increase in expenses due to ongoing care
            "one_time_cost": 10000.0, # Large upfront medical bill
            "investment_impact_pct": 0.0,
            "duration_months": 12,
            "message": "Simulates a large one-time medical bill and increased ongoing medical expenses."
        },
        "Market Downturn (20% loss)": {
            "income_reduction_pct": 0.0,
            "expense_increase_pct": 0.0,
            "one_time_cost": 0.0,
            "investment_impact_pct": 0.20, # 20% loss in investments
            "duration_months": 1, # Immediate impact
            "message": "Simulates an immediate 20% loss in investment portfolio value."
        },
        "Unexpected Large Expense": {
            "income_reduction_pct": 0.0,
            "expense_increase_pct": 0.0,
            "one_time_cost": 5000.0, # e.g., car repair, home repair deductible
            "investment_impact_pct": 0.0,
            "duration_months": 1, # Immediate impact
            "message": "Simulates an unexpected large one-time expense that must be paid immediately."
        }
    }

    profile = scenario_profiles.get(scenario_name, scenario_profiles["Job Loss (6 months)"])

    # Apply immediate impacts
    sim_income = monthly_income * (1 - profile["income_reduction_pct"])
    sim_expenses = monthly_expenses * (1 + profile["expense_increase_pct"])
    
    # Deduct one-time cost from liquid assets first, then investments
    remaining_one_time_cost = profile["one_time_cost"]
    if remaining_one_time_cost > 0:
        if initial_liquid_assets >= remaining_one_time_cost:
            initial_liquid_assets -= remaining_one_time_cost
            remaining_one_time_cost = 0
        else:
            remaining_one_time_cost -= initial_liquid_assets
            initial_liquid_assets = 0
            if current_investments >= remaining_one_time_cost:
                current_investments -= remaining_one_time_cost
                remaining_one_time_cost = 0
            else:
                current_investments = 0
                # Debt might increase here, but for simplicity, we'll just track asset depletion

    # Apply investment impact
    current_investments *= (1 - profile["investment_impact_pct"])

    # Simulation over time
    liquid_assets_history = [initial_liquid_assets]
    investments_history = [current_investments]
    net_worth_history = [initial_net_worth]
    
    current_liquid_assets = initial_liquid_assets
    current_investments_val = current_investments
    months_of_resilience = 0
    
    for month in range(simulation_months):
        # Calculate cash flow for the month
        if month < profile["duration_months"]:
            cash_flow = sim_income - sim_expenses
        else:
            cash_flow = monthly_income - sim_expenses # Income returns to normal after duration

        current_liquid_assets += cash_flow
        
        # If liquid assets run out, start drawing from investments
        if current_liquid_assets < 0:
            draw_from_investments = abs(current_liquid_assets)
            current_investments_val -= draw_from_investments
            current_liquid_assets = 0 # Reset liquid assets after drawing

        # Track resilience
        if current_liquid_assets >= 0 and current_investments_val >= 0:
            months_of_resilience += 1
        else:
            # If both liquid assets and investments are depleted, stop counting resilience
            pass # Keep tracking history, but resilience stops

        liquid_assets_history.append(current_liquid_assets)
        investments_history.append(current_investments_val)
        net_worth_history.append(current_liquid_assets + current_investments_val - total_debt) # Debt assumed constant for now

    final_liquid_assets = current_liquid_assets
    final_investments = current_investments_val
    final_net_worth = final_liquid_assets + final_investments - total_debt

    # Determine outcome
    if final_liquid_assets < 0 or final_investments < 0:
        outcome = "Severe Impact: Assets depleted, potential for debt increase."
        severity = "High"
    elif months_of_resilience < simulation_months:
        outcome = f"Moderate Impact: Assets lasted {months_of_resilience} months, but depleted before simulation end."
        severity = "Medium"
    else:
        outcome = "Low Impact: Assets remained positive throughout the simulation."
        severity = "Low"

    return {
        "scenario_name": scenario_name,
        "message": profile["message"],
        "initial_liquid_assets": initial_liquid_assets,
        "initial_investments": current_investments, # This is after initial impact
        "initial_net_worth": initial_net_worth,
        "final_liquid_assets": final_liquid_assets,
        "final_investments": final_investments,
        "final_net_worth": final_net_worth,
        "months_of_resilience": months_of_resilience,
        "outcome": outcome,
        "severity": severity,
        "liquid_assets_history": liquid_assets_history,
        "investments_history": investments_history,
        "net_worth_history": net_worth_history
    }


def calculate_wealth_projection(
    initial_capital: float,
    monthly_contribution: float,
    annual_return_rate: float,
    projection_years: int
) -> dict:
    """
    Projects wealth growth over time using compound interest.
    
    Args:
        initial_capital (float): Starting amount of money.
        monthly_contribution (float): Amount contributed each month.
        annual_return_rate (float): Annual rate of return (e.g., 0.07 for 7%).
        projection_years (int): Number of years to project wealth.
        
    Returns:
        dict: Contains final projected wealth and a history of wealth over time.
    """
    
    monthly_return_rate = (1 + annual_return_rate)**(1/12) - 1
    total_months = projection_years * 12
    
    wealth_history = [initial_capital]
    current_wealth = initial_capital
    
    for _ in range(total_months):
        current_wealth = (current_wealth + monthly_contribution) * (1 + monthly_return_rate)
        wealth_history.append(current_wealth)
        
    final_wealth = current_wealth
    
    return {
        "final_wealth": final_wealth,
        "wealth_history": wealth_history,
        "projection_years": projection_years
    }


def prepare_clustering_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data specifically for clustering analysis."""
    
    features = ['monthly_income', 'monthly_expenses', 'monthly_savings', 
                'monthly_investments', 'credit_score', 'total_debt', 
                'subscription_count', 'lifestyle_score']
    
    clustering_df = df[features].copy()
    
    # Handle any remaining NaN
    clustering_df = clustering_df.fillna(clustering_df.median())
    
    return clustering_df


def prepare_classification_data(df: pd.DataFrame) -> tuple:
    """Prepare features and target for classification."""
    
    feature_cols = ['age', 'monthly_income', 'monthly_expenses', 'monthly_savings',
                   'monthly_investments', 'credit_score', 'total_debt', 
                   'monthly_loan_payments', 'subscription_count', 'lifestyle_score',
                   'savings_rate', 'expense_ratio', 'debt_to_income']
    
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['risk_label'])
    
    return X, y, le


def prepare_regression_data(monthly_df: pd.DataFrame, users_df: pd.DataFrame = None) -> tuple:
    """Prepare features and target for regression (predicting savings)."""
    
    # Use monthly data features - income and expenses from monthly_df
    feature_cols = ['month', 'income', 'expenses', 'savings_rate']
    
    X = monthly_df[feature_cols].copy()
    X = X.fillna(X.median())
    
    y = monthly_df['savings'].values
    
    return X, y


if __name__ == "__main__":
    # Test preprocessing
    from data_generator import generate_full_dataset
    
    users_df, monthly_df = generate_full_dataset(100)
    
    preprocessor = FinancialDataPreprocessor()
    
    processed_users = preprocessor.preprocess_users(users_df)
    print("Processed Users Shape:", processed_users.shape)
    print("\nNew Features:")
    print(processed_users[['savings_rate', 'expense_ratio', 'stability_index']].head())
