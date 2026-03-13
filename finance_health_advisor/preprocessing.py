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


def prepare_regression_data(df: pd.DataFrame) -> tuple:
    """Prepare features and target for regression (predicting savings)."""
    
    feature_cols = ['age', 'monthly_income', 'monthly_expenses', 'credit_score',
                   'monthly_loan_payments', 'month', 'income', 'expenses',
                   'savings_rate']
    
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    y = df['savings'].values
    
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