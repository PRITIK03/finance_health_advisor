"""
Personalized Recommendations Engine
Provides actionable financial advice based on ML analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class RecommendationsEngine:
    """Generate personalized financial recommendations based on user profiles and ML predictions."""
    
    def __init__(self, users_df: pd.DataFrame, monthly_df: pd.DataFrame):
        """
        Initialize the recommendations engine.
        
        Args:
            users_df: DataFrame with user financial data
            monthly_df: DataFrame with monthly transaction data
        """
        self.users_df = users_df
        self.monthly_df = monthly_df
    
    def get_budget_recommendations(self, user_data: Dict) -> List[Dict]:
        """
        Generate budget allocation recommendations based on income and expenses.
        
        Uses the 50/30/20 rule as a baseline and customizes based on user profile.
        
        Args:
            user_data: Dictionary containing user financial information
            
        Returns:
            List of budget recommendation dictionaries
        """
        income = user_data.get('monthly_income', 0)
        expenses = user_data.get('monthly_expenses', 0)
        savings = user_data.get('monthly_savings', 0)
        savings_rate = (savings / income * 100) if income > 0 else 0
        
        recommendations = []
        
        # Calculate ideal budgets using 50/30/20 rule
        needs = income * 0.50  # 50% for needs
        wants = income * 0.30  # 30% for wants
        savings_target = income * 0.20  # 20% for savings
        
        # Current spending analysis
        current_needs = expenses * 0.80  # Assume 80% of expenses are needs
        current_wants = expenses * 0.20  # Assume 20% of expenses are wants
        
        # Needs recommendations
        if current_needs > needs:
            recommendations.append({
                'category': 'Needs',
                'status': 'warning',
                'message': f'Your needs spending (${current_needs:,.0f}) exceeds the recommended 50% (${needs:,.0f})',
                'suggestion': 'Consider reducing housing costs or utility expenses to bring needs within 50% of income.'
            })
        else:
            recommendations.append({
                'category': 'Needs',
                'status': 'good',
                'message': f'Your needs spending (${current_needs:,.0f}) is within the recommended 50% (${needs:,.0f})',
                'suggestion': 'Great job! Your essential expenses are well managed.'
            })
        
        # Wants recommendations
        if current_wants > wants:
            recommendations.append({
                'category': 'Wants',
                'status': 'warning',
                'message': f'Your wants spending (${current_wants:,.0f}) exceeds the recommended 30% (${wants:,.0f})',
                'suggestion': 'Consider the "24-Hour Rule" for non-essential purchases to reduce wants spending.'
            })
        else:
            recommendations.append({
                'category': 'Wants',
                'status': 'good',
                'message': f'Your wants spending (${current_wants:,.0f}) is within the recommended 30% (${wants:,.0f})',
                'suggestion': 'Well done! You have a good balance between needs and wants spending.'
            })
        
        # Savings recommendations
        if savings_rate < 20:
            recommendations.append({
                'category': 'Savings',
                'status': 'warning',
                'message': f'Your savings rate ({savings_rate:.1f}%) is below the recommended 20%',
                'suggestion': 'Try the "Pay Yourself First" method - automate savings transfers on payday.'
            })
        else:
            recommendations.append({
                'category': 'Savings',
                'status': 'good',
                'message': f'Your savings rate ({savings_rate:.1f}%) meets or exceeds the 20% target',
                'suggestion': 'Excellent! Consider increasing savings to 30% to accelerate wealth building.'
            })
        
        return recommendations
    
    def get_debt_recommendations(self, user_data: Dict) -> List[Dict]:
        """
        Generate debt payoff recommendations based on debt type and amount.
        
        Args:
            user_data: Dictionary containing user financial information
            
        Returns:
            List of debt recommendation dictionaries
        """
        debt = user_data.get('total_debt', 0)
        income = user_data.get('monthly_income', 0)
        credit_score = user_data.get('credit_score', 0)
        loan_payments = user_data.get('monthly_loan_payments', 0)
        
        recommendations = []
        
        if debt == 0:
            recommendations.append({
                'category': 'Debt-Free',
                'status': 'excellent',
                'message': 'You have no outstanding debt!',
                'suggestion': 'Focus on maximizing savings and investments to build wealth.'
            })
            return recommendations
        
        # Calculate debt-to-income ratio
        dti = (loan_payments / income * 100) if income > 0 else 0
        
        if dti > 36:
            recommendations.append({
                'category': 'Debt-to-Income',
                'status': 'critical',
                'message': f'Your debt-to-income ratio ({dti:.1f}%) exceeds the recommended 36%',
                'suggestion': 'Prioritize paying down high-interest debt. Consider debt consolidation.'
            })
        elif dti > 20:
            recommendations.append({
                'category': 'Debt-to-Income',
                'status': 'warning',
                'message': f'Your debt-to-income ratio ({dti:.1f}%) is moderate (20-36%)',
                'suggestion': 'Consider the Avalanche Method - pay minimums on all debts, extra on highest interest.'
            })
        else:
            recommendations.append({
                'category': 'Debt-to-Income',
                'status': 'good',
                'message': f'Your debt-to-income ratio ({dti:.1f}%) is healthy',
                'suggestion': 'Your debt levels are manageable. Continue making consistent payments.'
            })
        
        # Credit score recommendations
        if credit_score < 600:
            recommendations.append({
                'category': 'Credit Score',
                'status': 'warning',
                'message': f'Your credit score ({credit_score}) needs improvement',
                'suggestion': 'Pay all bills on time, reduce credit utilization below 30%, and dispute any errors on your credit report.'
            })
        elif credit_score < 700:
            recommendations.append({
                'category': 'Credit Score',
                'status': 'moderate',
                'message': f'Your credit score ({credit_score}) is fair',
                'suggestion': 'Focus on paying down debt and keeping credit utilization low to reach "Good" status (700+).'
            })
        else:
            recommendations.append({
                'category': 'Credit Score',
                'status': 'good',
                'message': f'Your credit score ({credit_score}) is good or excellent',
                'suggestion': 'Maintain your good credit by continuing timely payments and low utilization.'
            })
        
        # Debt payoff strategy
        recommendations.append({
            'category': 'Strategy',
            'status': 'info',
            'message': 'Recommended Debt Payoff Approach',
            'suggestion': 'Use the Snowball Method (pay smallest debts first) for motivation, or Avalanche Method (pay highest interest first) to save money.'
        })
        
        return recommendations
    
    def get_savings_recommendations(self, user_data: Dict) -> List[Dict]:
        """
        Generate savings and emergency fund recommendations.
        
        Args:
            user_data: Dictionary containing user financial information
            
        Returns:
            List of savings recommendation dictionaries
        """
        income = user_data.get('monthly_income', 0)
        savings = user_data.get('monthly_savings', 0)
        expenses = user_data.get('monthly_expenses', 0)
        
        recommendations = []
        
        # Emergency fund calculation
        monthly_expenses = expenses
        recommended_emergency_fund = monthly_expenses * 6  # 6 months recommended
        
        # Get user's current savings (approximate from data)
        current_savings = savings * 12  # Annual savings estimate
        
        if current_savings < recommended_emergency_fund * 0.5:
            recommendations.append({
                'category': 'Emergency Fund',
                'status': 'critical',
                'message': f'Your emergency fund is below 50% of the recommended ${recommended_emergency_fund:,.0f}',
                'suggestion': 'Prioritize building a 3-month emergency fund before other financial goals. Set up automatic transfers.'
            })
        elif current_savings < recommended_emergency_fund:
            recommendations.append({
                'category': 'Emergency Fund',
                'status': 'warning',
                'message': f'Your emergency fund covers {current_savings/recommended_emergency_fund*100:.0f}% of the ${recommended_emergency_fund:,.0f} target',
                'suggestion': 'Continue building toward a full 6-month emergency fund. You\'re making progress!'
            })
        else:
            recommendations.append({
                'category': 'Emergency Fund',
                'status': 'good',
                'message': f'You have a healthy emergency fund (${current_savings:,.0f})',
                'suggestion': 'Consider investing excess emergency funds in a high-yield savings account or money market fund.'
            })
        
        # Retirement savings
        age = user_data.get('age', 30)
        retirement_target = income * 12 * 20  # 20x annual income target
        years_to_retirement = 65 - age if age < 65 else 0
        
        if years_to_retirement > 0:
            monthly_contribution_needed = (retirement_target - current_savings) / (years_to_retirement * 12)
            
            if monthly_contribution_needed > income * 0.30:
                recommendations.append({
                    'category': 'Retirement',
                    'status': 'warning',
                    'message': f'Based on your age ({age}), you need to save ${monthly_contribution_needed:,.0f}/month for retirement',
                    'suggestion': 'Consider maximizing 401(k) match, opening an IRA, or increasing retirement contributions gradually.'
                })
            else:
                recommendations.append({
                    'category': 'Retirement',
                    'status': 'good',
                    'message': f'You need to save ${monthly_contribution_needed:,.0f}/month to reach retirement goals',
                    'suggestion': 'You\'re on track! Continue current savings rate and consider maximizing tax-advantaged accounts.'
                })
        else:
            recommendations.append({
                'category': 'Retirement',
                'status': 'info',
                'message': 'Approaching retirement age',
                'suggestion': 'Review your retirement accounts and consider consulting a financial advisor for distribution strategies.'
            })
        
        return recommendations
    
    def get_investment_recommendations(self, user_data: Dict) -> List[Dict]:
        """
        Generate investment recommendations based on risk profile.
        
        Args:
            user_data: Dictionary containing user financial information
            
        Returns:
            List of investment recommendation dictionaries
        """
        income = user_data.get('monthly_income', 0)
        risk_label = user_data.get('risk_label', 'Medium')
        age = user_data.get('age', 30)
        financial_health = user_data.get('financial_health_score', 50)
        
        recommendations = []
        
        diversification_profile = self.get_diversification_profile(user_data)
        risk_tolerance = diversification_profile['risk_tolerance']
        allocations = diversification_profile['target_allocation']
        strategy = diversification_profile['strategy']
        
        recommendations.append({
            'category': 'Asset Allocation',
            'status': 'info',
            'message': f'Risk Tolerance: {risk_tolerance}',
            'suggestion': f'Recommended allocation: Stocks {allocations["Stocks"]}%, Bonds {allocations["Bonds"]}%, Cash {allocations["Cash"]}%. {strategy}'
        })

        recommendations.append({
            'category': 'Diversification Score',
            'status': 'good' if diversification_profile['diversification_score'] >= 75 else 'warning',
            'message': f'Portfolio diversification score: {diversification_profile["diversification_score"]:.0f}/100',
            'suggestion': diversification_profile['rebalance_hint']
        })
        
        # Specific investment suggestions based on risk
        if risk_tolerance in ['Conservative', 'Moderate Conservative']:
            recommendations.append({
                'category': 'Investment Types',
                'status': 'info',
                'message': 'Recommended Investment Vehicles',
                'suggestion': 'Consider: Index funds, Bond ETFs, Dividend aristocrats, Money market funds, CDs.'
            })
        elif risk_tolerance == 'Moderate':
            recommendations.append({
                'category': 'Investment Types',
                'status': 'info',
                'message': 'Recommended Investment Vehicles',
                'suggestion': 'Consider: S&P 500 index fund, Balanced mutual funds, REITs, International diversification.'
            })
        else:
            recommendations.append({
                'category': 'Investment Types',
                'status': 'info',
                'message': 'Recommended Investment Vehicles',
                'suggestion': 'Consider: Growth ETFs, Small-cap index funds, International emerging markets, Sector-specific investments.'
            })
        
        # Tax-advantaged accounts
        recommendations.append({
            'category': 'Tax Strategy',
            'status': 'good',
            'message': 'Tax-Advantaged Account Recommendations',
            'suggestion': 'Maximize 401(k) contributions (especially employer match), Roth IRA, and HSAs before taxable accounts.'
        })

        # Tax Optimization (New Feature)
        annual_income = income * 12
        if annual_income > 50000:
            recommendations.append({
                'category': 'Tax Strategy',
                'status': 'info',
                'message': f'With an annual income of ${annual_income:,.0f}, you are in a higher tax bracket.',
                'suggestion': 'Maximize your 401(k) and HSA contributions to reduce your taxable income.'
            })
        
        return recommendations

    def get_diversification_profile(self, user_data: Dict) -> Dict:
        """Create a simple current-vs-target diversification profile for the UI."""
        age = user_data.get('age', 30)
        risk_label = user_data.get('risk_label', 'Medium')
        financial_health = user_data.get('financial_health_score', 50)
        income = user_data.get('monthly_income', 0)
        investment_rate = (user_data.get('monthly_investments', 0) / income) if income > 0 else 0
        debt_ratio = (user_data.get('total_debt', 0) / max(income * 12, 1))

        risk_tolerance = self._calculate_risk_tolerance(age, risk_label, financial_health)
        target_allocation = self._get_target_allocation(risk_tolerance)

        if risk_tolerance == 'Conservative':
            current_allocation = {'Stocks': 35, 'Bonds': 40, 'Cash': 15, 'Real Estate': 5, 'Crypto': 0, 'Alternatives': 5}
        elif risk_tolerance == 'Moderate Conservative':
            current_allocation = {'Stocks': 48, 'Bonds': 28, 'Cash': 12, 'Real Estate': 7, 'Crypto': 1, 'Alternatives': 4}
        elif risk_tolerance == 'Moderate':
            current_allocation = {'Stocks': 58, 'Bonds': 18, 'Cash': 10, 'Real Estate': 8, 'Crypto': 2, 'Alternatives': 4}
        elif risk_tolerance == 'Moderate Aggressive':
            current_allocation = {'Stocks': 68, 'Bonds': 12, 'Cash': 7, 'Real Estate': 7, 'Crypto': 3, 'Alternatives': 3}
        else:
            current_allocation = {'Stocks': 78, 'Bonds': 7, 'Cash': 5, 'Real Estate': 5, 'Crypto': 3, 'Alternatives': 2}

        if investment_rate < 0.08:
            current_allocation['Cash'] += 5
            current_allocation['Stocks'] -= 5
        if debt_ratio > 0.35:
            current_allocation['Cash'] += 3
            current_allocation['Crypto'] = max(0, current_allocation['Crypto'] - 2)
            current_allocation['Stocks'] -= 1

        categories = list(current_allocation.keys())
        absolute_gap = sum(abs(current_allocation[k] - target_allocation[k]) for k in categories)
        diversification_score = max(0.0, 100 - (absolute_gap / 2))
        gap_by_asset = {k: round(target_allocation[k] - current_allocation[k], 1) for k in categories}
        largest_gap_asset = max(gap_by_asset, key=lambda x: abs(gap_by_asset[x]))
        largest_gap = gap_by_asset[largest_gap_asset]

        if largest_gap > 0:
            rebalance_hint = f"Consider increasing {largest_gap_asset} exposure by about {abs(largest_gap):.0f}% over time."
        elif largest_gap < 0:
            rebalance_hint = f"Consider trimming {largest_gap_asset} exposure by about {abs(largest_gap):.0f}% and reallocating gradually."
        else:
            rebalance_hint = "Your portfolio is reasonably aligned with the target allocation."

        return {
            'risk_tolerance': risk_tolerance,
            'target_allocation': target_allocation,
            'current_allocation': current_allocation,
            'diversification_score': diversification_score,
            'gap_by_asset': gap_by_asset,
            'largest_gap_asset': largest_gap_asset,
            'rebalance_hint': rebalance_hint,
            'strategy': self._get_allocation_strategy(risk_tolerance)
        }

    def _get_target_allocation(self, risk_tolerance: str) -> Dict[str, int]:
        """Return target allocation weights by risk tolerance."""
        targets = {
            'Conservative': {'Stocks': 30, 'Bonds': 45, 'Cash': 15, 'Real Estate': 5, 'Crypto': 0, 'Alternatives': 5},
            'Moderate Conservative': {'Stocks': 45, 'Bonds': 30, 'Cash': 10, 'Real Estate': 8, 'Crypto': 2, 'Alternatives': 5},
            'Moderate': {'Stocks': 60, 'Bonds': 20, 'Cash': 8, 'Real Estate': 7, 'Crypto': 2, 'Alternatives': 3},
            'Moderate Aggressive': {'Stocks': 72, 'Bonds': 12, 'Cash': 6, 'Real Estate': 6, 'Crypto': 2, 'Alternatives': 2},
            'Aggressive': {'Stocks': 82, 'Bonds': 6, 'Cash': 4, 'Real Estate': 4, 'Crypto': 2, 'Alternatives': 2}
        }
        return targets.get(risk_tolerance, targets['Moderate'])

    def _get_allocation_strategy(self, risk_tolerance: str) -> str:
        """Return a short strategy description for the target mix."""
        strategies = {
            'Conservative': 'Focus on stability, income, and lower drawdowns.',
            'Moderate Conservative': 'Blend capital preservation with measured long-term growth.',
            'Moderate': 'Balance growth and resilience across multiple asset classes.',
            'Moderate Aggressive': 'Lean toward long-term growth while keeping some stabilizers.',
            'Aggressive': 'Prioritize long-term capital appreciation and accept higher volatility.'
        }
        return strategies.get(risk_tolerance, strategies['Moderate'])
    
    def _calculate_risk_tolerance(self, age: int, risk_label: str, health_score: int) -> str:
        """
        Calculate risk tolerance based on age, risk label, and financial health score.
        
        Args:
            age: User's age
            risk_label: ML-predicted risk category
            health_score: Financial health score
            
        Returns:
            Risk tolerance category string
        """
        # Base risk tolerance on age (100 - age rule)
        age_based_risk = 100 - age
        
        # Adjust based on ML risk prediction
        risk_adjustments = {
            'Very Low': 20,
            'Low': 10,
            'Medium': 0,
            'High': -10,
            'Very High': -20
        }
        
        adjustment = risk_adjustments.get(risk_label, 0)
        
        # Adjust based on financial health
        health_adjustment = (health_score - 50) / 5
        
        final_risk = age_based_risk + adjustment + health_adjustment
        
        if final_risk >= 80:
            return 'Aggressive'
        elif final_risk >= 60:
            return 'Moderate Aggressive'
        elif final_risk >= 40:
            return 'Moderate'
        elif final_risk >= 20:
            return 'Moderate Conservative'
        else:
            return 'Conservative'
    
    def get_all_recommendations(self, user_data: Dict) -> Dict[str, List[Dict]]:
        """
        Get comprehensive recommendations across all categories.
        
        Args:
            user_data: Dictionary containing user financial information
            
        Returns:
            Dictionary with all recommendation categories
        """
        return {
            'budget': self.get_budget_recommendations(user_data),
            'debt': self.get_debt_recommendations(user_data),
            'savings': self.get_savings_recommendations(user_data),
            'investments': self.get_investment_recommendations(user_data)
        }
    
    def generate_user_recommendations(self, user_id: int) -> Dict[str, List[Dict]]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dictionary with all recommendation categories for the user
        """
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        
        if user_row.empty:
            return {
                'budget': [],
                'debt': [],
                'savings': [],
                'investments': []
            }
        
        user_data = user_row.iloc[0].to_dict()
        
        return self.get_all_recommendations(user_data)
    
    def generate_cohort_recommendations(self, cluster: int = None, risk_label: str = None) -> Dict[str, any]:
        """
        Generate aggregate recommendations for a user cohort.
        
        Args:
            cluster: Optional cluster filter
            risk_label: Optional risk label filter
            
        Returns:
            Dictionary with cohort-level recommendations and statistics
        """
        cohort = self.users_df.copy()
        
        if cluster is not None:
            cohort = cohort[cohort['cluster'] == cluster]
        if risk_label is not None:
            cohort = cohort[cohort['risk_label'] == risk_label]
        
        if cohort.empty:
            return {
                'cohort_size': 0,
                'recommendations': {},
                'summary': 'No users found in the specified cohort.'
            }
        
        # Calculate cohort averages
        avg_income = cohort['monthly_income'].mean()
        avg_expenses = cohort['monthly_expenses'].mean()
        avg_savings = cohort['monthly_savings'].mean()
        avg_health = cohort['financial_health_score'].mean()
        
        # Get most common risk label
        common_risk = cohort['risk_label'].mode().iloc[0]
        
        # Generate representative user data for recommendations
        representative_user = {
            'monthly_income': avg_income,
            'monthly_expenses': avg_expenses,
            'monthly_savings': avg_savings,
            'total_debt': cohort['total_debt'].mean(),
            'monthly_loan_payments': cohort['monthly_loan_payments'].mean(),
            'credit_score': cohort['credit_score'].mean(),
            'age': cohort['age'].mean(),
            'risk_label': common_risk,
            'financial_health_score': avg_health
        }
        
        recommendations = self.get_all_recommendations(representative_user)
        
        return {
            'cohort_size': len(cohort),
            'cohort_stats': {
                'avg_income': avg_income,
                'avg_expenses': avg_expenses,
                'avg_savings': avg_savings,
                'avg_health_score': avg_health,
                'common_risk_label': common_risk
            },
            'recommendations': recommendations,
            'summary': self._generate_cohort_summary(len(cohort), avg_health, common_risk)
        }
    
    def _generate_cohort_summary(self, size: int, avg_health: float, common_risk: str) -> str:
        """Generate a summary message for the cohort."""
        return f"""
        This cohort contains {size:,} users with an average financial health score of {avg_health:.1f}/100.
        Most users ({common_risk} risk) would benefit from the personalized recommendations provided above.
        """
