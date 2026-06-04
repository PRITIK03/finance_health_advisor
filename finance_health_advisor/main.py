"""
Main Entry Point for Finance Health Advisor
Runs the complete ML pipeline and generates visualizations
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from logging_config import setup_logger
from data_generator import generate_full_dataset
from preprocessing import FinancialDataPreprocessor
from models import train_all_models
from visualizations import FinancialVisualizer, generate_summary_statistics


# Setup logger
logger = setup_logger("fha", CONFIG.log_level)


def main():
    """Run the complete finance health advisor pipeline."""

    logger.info("=" * 70)
    logger.info("   AI-POWERED PERSONAL FINANCE HEALTH ADVISOR")
    logger.info("   End-to-End Machine Learning Data Analysis")
    logger.info("=" * 70)

    # Step 1: Generate data
    logger.info("\n[STEP 1] Generating synthetic financial data...")
    users_df, monthly_df = generate_full_dataset(CONFIG.data.total_users)
    logger.info(f"   - Generated {len(users_df):,} user profiles")
    logger.info(f"   - Generated {len(monthly_df):,} monthly records")

    # Step 2: Preprocess data
    logger.info("\n[STEP 2] Preprocessing data...")
    preprocessor = FinancialDataPreprocessor()
    users_processed = preprocessor.preprocess_users(users_df)
    monthly_processed = preprocessor.preprocess_monthly(monthly_df)
    logger.info("   - Encoded categorical features")
    logger.info("   - Engineered derived features")
    logger.info("   - Handled missing values")

    # Step 3: Train ML models
    logger.info("\n[STEP 3] Training Machine Learning models...")
    results, pipeline = train_all_models(users_df, monthly_df)

    # Step 4: Generate visualizations
    logger.info("\n[STEP 4] Generating visualizations...")
    visualizer = FinancialVisualizer(users_processed, monthly_processed)
    viz_dict = visualizer.create_all_visualizations()
    logger.info(f"   - Created {len(viz_dict)} visualizations")

    # Step 5: Generate summary statistics
    logger.info("\n[STEP 5] Generating summary statistics...")
    stats = generate_summary_statistics(users_processed, monthly_processed)

    logger.info("\n" + "=" * 70)
    logger.info("   PIPELINE COMPLETE!")
    logger.info("=" * 70)

    logger.info("\nSUMMARY STATISTICS:")
    logger.info(f"   * Total Users: {stats['total_users']:,}")
    logger.info(f"   * Total Monthly Records: {stats['total_monthly_records']:,}")
    logger.info(f"   * Average Income: ${stats['avg_income']:,.2f}")
    logger.info(f"   * Average Expenses: ${stats['avg_expenses']:,.2f}")
    logger.info(f"   * Average Savings: ${stats['avg_savings']:,.2f}")
    logger.info(f"   * Average Health Score: {stats['avg_health_score']:.2f}/100")
    logger.info(f"   * Average Credit Score: {stats['avg_credit_score']:.0f}")

    logger.info("\nRISK DISTRIBUTION:")
    for risk, count in stats['risk_distribution'].items():
        pct = count / stats['total_users'] * 100
        logger.info(f"   * {risk}: {count:,} ({pct:.1f}%)")

    logger.info("\nML MODEL PERFORMANCE:")
    logger.info(f"   * Clustering Silhouette Score: {results.get('silhouette', 'N/A')}")
    logger.info(f"   * Classification Accuracy: {results['classification']['test_metrics']['accuracy']:.2%}")
    logger.info(f"   * Forecasting R2 Score: {results['forecasting']['test_metrics']['r2']:.4f}")

    logger.info("\nTO RUN THE INTERACTIVE DASHBOARD:")
    logger.info("   Run: streamlit run app.py")

    logger.info("\n" + "=" * 70)

    return users_df, monthly_df, results


if __name__ == "__main__":
    users_df, monthly_df, results = main()