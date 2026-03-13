"""
Main Entry Point for Finance Health Advisor
Runs the complete ML pipeline and generates visualizations
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_full_dataset
from preprocessing import FinancialDataPreprocessor
from models import train_all_models
from visualizations import FinancialVisualizer, generate_summary_statistics


def main():
    """Run the complete finance health advisor pipeline."""
    
    print("=" * 70)
    print("   AI-POWERED PERSONAL FINANCE HEALTH ADVISOR")
    print("   End-to-End Machine Learning Data Analysis")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\n[STEP 1] Generating synthetic financial data...")
    users_df, monthly_df = generate_full_dataset(10000)
    print(f"   - Generated {len(users_df):,} user profiles")
    print(f"   - Generated {len(monthly_df):,} monthly records")
    
    # Step 2: Preprocess data
    print("\n[STEP 2] Preprocessing data...")
    preprocessor = FinancialDataPreprocessor()
    users_processed = preprocessor.preprocess_users(users_df)
    monthly_processed = preprocessor.preprocess_monthly(monthly_df)
    print("   - Encoded categorical features")
    print("   - Engineered derived features")
    print("   - Handled missing values")
    
    # Step 3: Train ML models
    print("\n[STEP 3] Training Machine Learning models...")
    results = train_all_models(users_df, monthly_df)
    
    # Step 4: Generate visualizations
    print("\n[STEP 4] Generating visualizations...")
    visualizer = FinancialVisualizer(users_df, monthly_df)
    viz_dict = visualizer.create_all_visualizations()
    print(f"   - Created {len(viz_dict)} visualizations")
    
    # Step 5: Generate summary statistics
    print("\n[STEP 5] Generating summary statistics...")
    stats = generate_summary_statistics(users_df, monthly_df)
    
    print("\n" + "=" * 70)
    print("   PIPELINE COMPLETE!")
    print("=" * 70)
    
    print("\nSUMMARY STATISTICS:")
    print(f"   * Total Users: {stats['total_users']:,}")
    print(f"   * Total Monthly Records: {stats['total_monthly_records']:,}")
    print(f"   * Average Income: ${stats['avg_income']:,.2f}")
    print(f"   * Average Expenses: ${stats['avg_expenses']:,.2f}")
    print(f"   * Average Savings: ${stats['avg_savings']:,.2f}")
    print(f"   * Average Health Score: {stats['avg_health_score']:.2f}/100")
    print(f"   * Average Credit Score: {stats['avg_credit_score']:.0f}")
    
    print("\nRISK DISTRIBUTION:")
    for risk, count in stats['risk_distribution'].items():
        pct = count / stats['total_users'] * 100
        print(f"   * {risk}: {count:,} ({pct:.1f}%)")
    
    print("\nML MODEL PERFORMANCE:")
    print(f"   * Clustering Silhouette Score: {results.get('silhouette', 'N/A')}")
    print(f"   * Classification Accuracy: {results['classification']['test_metrics']['accuracy']:.2%}")
    print(f"   * Forecasting R2 Score: {results['forecasting']['test_metrics']['r2']:.4f}")
    
    print("\nTO RUN THE INTERACTIVE DASHBOARD:")
    print("   Run: streamlit run app.py")
    
    print("\n" + "=" * 70)
    
    return users_df, monthly_df, results


if __name__ == "__main__":
    users_df, monthly_df, results = main()