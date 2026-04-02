"""
Data Preprocessing and Feature Engineering Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


class FinancialDataPreprocessor:
    """Preprocess and engineer features for financial data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
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


def calculate_emergency_fund_stress_test(user_row: pd.Series, crisis_type: str = "Job Loss") -> dict:
    """
    Simulate how many months the current emergency fund lasts under different crises.
    crisis_type: 'Job Loss', 'Medical Emergency', 'Major Repair'
    """
    current_savings = user_row['monthly_savings'] * 6 # Assumption: 6 months of savings is the fund
    monthly_expenses = user_row['monthly_expenses']
    
    # Adjust expenses based on crisis
    if crisis_type == "Job Loss":
        # Cut discretionary spending (30% reduction)
        crisis_expenses = monthly_expenses * 0.7
        impact_message = "Reduced discretionary spending by 30%."
    elif crisis_type == "Medical Emergency":
        # Immediate one-time cost + slightly higher monthly
        current_savings -= 5000 # One-time $5k medical bill
        crisis_expenses = monthly_expenses * 1.1 # 10% more for follow-ups
        impact_message = "Paid $5,000 upfront + 10% monthly increase."
    else: # Major Repair
        current_savings -= 3000
        crisis_expenses = monthly_expenses
        impact_message = "Paid $3,000 one-time repair cost."
        
    months_buffer = max(0, current_savings / crisis_expenses) if crisis_expenses > 0 else 100
    
    return {
        'months_buffer': months_buffer,
        'crisis_expenses': crisis_expenses,
        'initial_fund': user_row['monthly_savings'] * 6,
        'remaining_fund_start': current_savings,
        'impact_message': impact_message,
        'is_safe': months_buffer >= 6
    }


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