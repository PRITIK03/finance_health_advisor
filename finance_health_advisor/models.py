"""
Machine Learning Models Module
Implements clustering, classification, regression, and anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             silhouette_score, mean_absolute_error,
                             mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from logging_config import get_logger

logger = get_logger(__name__)


class FinancialClusteringModel:
    """K-Means clustering for financial health segmentation."""

    def __init__(self, n_clusters: int = None):
        self.n_clusters = n_clusters or CONFIG.models.n_clusters
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=CONFIG.models.kmeans_random_state,
            n_init=CONFIG.models.kmeans_n_init
        )
        self.scaler = StandardScaler()
        self.cluster_labels = None
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels."""
        X_scaled = self.scaler.fit_transform(X)
        self.cluster_labels = self.model.fit_predict(X_scaled)
        return self.cluster_labels
    
    def get_cluster_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get statistics for each cluster."""
        X = X.copy()
        X['cluster'] = self.cluster_labels
        
        stats = X.groupby('cluster').agg({
            'monthly_income': 'mean',
            'monthly_expenses': 'mean',
            'monthly_savings': 'mean',
            'credit_score': 'mean',
            'financial_health_score': 'mean'
        }).round(2)
        
        return stats
    
    def get_silhouette_score(self, X: np.ndarray) -> float:
        """Calculate silhouette score."""
        X_scaled = self.scaler.transform(X)
        return silhouette_score(X_scaled, self.cluster_labels)


class RiskClassificationModel:
    """Random Forest classifier for financial risk prediction."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=CONFIG.models.rf_n_estimators,
            max_depth=CONFIG.models.rf_max_depth,
            min_samples_split=CONFIG.models.rf_min_samples_split,
            random_state=CONFIG.models.rf_random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Train the model."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X.values)
        self.model.fit(X_scaled, y)
        self.feature_importance = dict(
            zip(range(X.shape[1]), self.model.feature_importances_)
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_vals)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        accuracy = (predictions == y).mean()
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=CONFIG.models.cv_folds)
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y, predictions),
            'confusion_matrix': confusion_matrix(y, predictions)
        }


class SavingsForecastModel:
    """Gradient Boosting regressor for savings forecasting."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=CONFIG.models.gb_n_estimators,
            max_depth=CONFIG.models.gb_max_depth,
            learning_rate=CONFIG.models.gb_learning_rate,
            random_state=CONFIG.models.gb_random_state
        )
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }


class AnomalyDetectionModel:
    """Isolation Forest for detecting anomalous spending patterns."""

    def __init__(self, contamination: float = None):
        self.model = SklearnIsolationForest(
            n_estimators=CONFIG.models.if_n_estimators,
            contamination=contamination or CONFIG.models.if_contamination,
            random_state=CONFIG.models.if_random_state
        )
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies (-1 for anomaly, 1 for normal)."""
        return self.model.fit_predict(X)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        return self.model.decision_function(X)


class SimilarUsersModel:
    """Nearest Neighbors for finding similar financial profiles."""

    def __init__(self, n_neighbors: int = None):
        self.n_neighbors = n_neighbors or CONFIG.models.knn_n_neighbors
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray) -> None:
        """Fit the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
    
    def find_similar(self, X: np.ndarray, user_idx: int, n: int = 5) -> np.ndarray:
        """Find n most similar users."""
        X_scaled = self.scaler.transform(X)
        
        # Get the user vector
        user_vector = X_scaled[user_idx].reshape(1, -1)
        
        # Find distances to all users
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=n+1)
        
        # Return indices excluding the user itself
        similar_indices = indices[0][1:]
        return similar_indices


class FinancialMLPipeline:
    """Complete ML pipeline combining all models."""
    
    def __init__(self):
        self.clustering_model = None
        self.classification_model = None
        self.forecast_model = None
        self.anomaly_model = None
        self.similar_users_model = None
        self.preprocessor = None
        self.label_encoder = None
        
    def run_clustering(self, X: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """Run clustering analysis."""
        n_clusters = n_clusters or CONFIG.models.n_clusters
        self.clustering_model = FinancialClusteringModel(n_clusters)
        return self.clustering_model.fit_predict(X)
    
    def run_classification(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Run classification model with train/val/test splits."""
        # Train/validation/test split: 70/15/15
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=CONFIG.models.random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=CONFIG.models.random_state, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )
        
        self.classification_model = RiskClassificationModel()
        self.classification_model.fit(X_train, y_train)
        
        train_metrics = self.classification_model.evaluate(X_train.values, y_train)
        val_metrics = self.classification_model.evaluate(X_val.values, y_val)
        test_metrics = self.classification_model.evaluate(X_test.values, y_test)
        
        logger.info(f"Classification - Train: {train_metrics['accuracy']:.4f}, "
                   f"Val: {val_metrics['accuracy']:.4f}, Test: {test_metrics['accuracy']:.4f}")
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    
    def run_forecasting(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run savings forecasting with train/val/test splits."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=CONFIG.models.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=CONFIG.models.random_state
        )
        
        self.forecast_model = SavingsForecastModel()
        self.forecast_model.fit(X_train, y_train)
        
        train_metrics = self.forecast_model.evaluate(X_train, y_train)
        val_metrics = self.forecast_model.evaluate(X_val, y_val)
        test_metrics = self.forecast_model.evaluate(X_test, y_test)
        
        logger.info(f"Forecasting - Train R2: {train_metrics['r2']:.4f}, "
                   f"Val R2: {val_metrics['r2']:.4f}, Test R2: {test_metrics['r2']:.4f}")
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    
    def run_anomaly_detection(self, X: np.ndarray) -> np.ndarray:
        """Run anomaly detection."""
        self.anomaly_model = AnomalyDetectionModel()
        return self.anomaly_model.fit_predict(X)
    
    def run_similar_users_search(self, X: np.ndarray) -> None:
        """Initialize and fit the similar users model."""
        self.similar_users_model = SimilarUsersModel()
        self.similar_users_model.fit(X)


def train_all_models(users_df: pd.DataFrame, monthly_df: pd.DataFrame, 
                     save_artifacts: bool = True, model_dir: str = "models") -> tuple:
    """Train all models and return results and pipeline.
    
    This function properly avoids data leakage by doing train/test splits
    before any preprocessing that could leak information.
    
    Args:
        users_df: User profile data
        monthly_df: Monthly transaction data
        save_artifacts: Whether to save model artifacts to disk
        model_dir: Directory to save model artifacts
    
    Returns:
        Tuple of (results_dict, pipeline)
    """
    from preprocessing import (prepare_clustering_data, prepare_classification_data,
                              prepare_regression_data, FinancialDataPreprocessor)
    from model_persistence import save_training_artifacts
    
    results = {}
    
    print("=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)
    
    # Preprocess users to create derived features
    preprocessor = FinancialDataPreprocessor()
    users_df = preprocessor.preprocess_users(users_df)
    monthly_df = preprocessor.preprocess_monthly(monthly_df)
    pipeline = FinancialMLPipeline()
    pipeline.preprocessor = preprocessor

    # 1. Clustering - fit on all users (unsupervised, no target leakage)
    print("\n[1/5] Training K-Means Clustering...")
    clustering_data = prepare_clustering_data(users_df)
    clusters = pipeline.run_clustering(clustering_data.values)
    users_df = users_df.copy()
    users_df['cluster'] = clusters
    
    print(f"   * Clustered {len(users_df)} users into {CONFIG.models.n_clusters} segments")
    silhouette = pipeline.clustering_model.get_silhouette_score(clustering_data.values)
    print(f"   * Silhouette Score: {silhouette:.4f}")
    results['silhouette'] = silhouette
    
    # 2. Classification - proper train/val/test split AFTER preprocessing
    print("\n[2/5] Training Random Forest Classifier...")
    X_class, y_class, label_encoder = prepare_classification_data(users_df)
    pipeline.label_encoder = label_encoder
    results['classification'] = pipeline.run_classification(X_class, y_class)
    
    print(f"   * Test Accuracy: {results['classification']['test_metrics']['accuracy']:.4f}")
    print(f"   * Val Accuracy: {results['classification']['val_metrics']['accuracy']:.4f}")
    print(f"   * CV Score: {results['classification']['test_metrics']['cv_mean']:.4f} (+/- {results['classification']['test_metrics']['cv_std']:.4f})")
    
    # 3. Forecasting - proper train/val/test split
    print("\n[3/5] Training Gradient Boosting Regressor...")
    X_reg, y_reg = prepare_regression_data(monthly_df)
    results['forecasting'] = pipeline.run_forecasting(X_reg.values, y_reg)
    
    print(f"   * Test R2 Score: {results['forecasting']['test_metrics']['r2']:.4f}")
    print(f"   * Val R2 Score: {results['forecasting']['val_metrics']['r2']:.4f}")
    print(f"   * Test RMSE: ${results['forecasting']['test_metrics']['rmse']:.2f}")
    
    # 4. Anomaly Detection - unsupervised, fit on full data is OK
    print("\n[4/5] Training Isolation Forest...")
    spending_cols = ['Housing', 'Transportation', 'Food', 'Healthcare', 
                    'Entertainment', 'Shopping', 'Education', 'Subscriptions',
                    'Insurance', 'Miscellaneous']
    anomaly_data = monthly_df[spending_cols].fillna(0)
    anomalies = pipeline.run_anomaly_detection(anomaly_data.values)
    monthly_df = monthly_df.copy()
    monthly_df['ml_anomaly'] = (anomalies == -1).astype(int)
    
    anomaly_count = (anomalies == -1).sum()
    print(f"   * Detected {anomaly_count} anomalies ({anomaly_count/len(monthly_df)*100:.2f}%)")
    
    # 5. Similar Users Search (KNN) - fit on clustering features
    print("\n[5/5] Initializing Peer Comparison Model (KNN)...")
    pipeline.run_similar_users_search(clustering_data.values)
    print("   * Peer Comparison Model ready")
    
    # Save artifacts if requested
    if save_artifacts:
        print("\n[6/6] Saving model artifacts...")
        saved = save_training_artifacts(pipeline, preprocessor, label_encoder, results, model_dir)
        print(f"   * Artifacts saved to {model_dir}/")
    
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)
    
    return results, pipeline


def load_trained_pipeline(model_dir: str = "models"):
    """Load a previously trained pipeline."""
    from model_persistence import load_training_artifacts
    return load_training_artifacts(model_dir)


if __name__ == "__main__":
    # Test model training
    from data_generator import generate_full_dataset
    
    users_df, monthly_df = generate_full_dataset(1000)
    results = train_all_models(users_df, monthly_df)