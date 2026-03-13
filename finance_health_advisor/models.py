"""
Machine Learning Models Module
Implements clustering, classification, regression, and anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             silhouette_score, mean_absolute_error, 
                             mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')


class FinancialClusteringModel:
    """K-Means clustering for financial health segmentation."""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.feature_importance = dict(
            zip(range(X.shape[1]), self.model.feature_importances_)
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
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
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
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
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
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
    
    def __init__(self, contamination: float = 0.02):
        self.model = SklearnIsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies (-1 for anomaly, 1 for normal)."""
        return self.model.fit_predict(X)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        return self.model.decision_function(X)


class SimilarUsersModel:
    """K-Nearest Neighbors for finding similar financial profiles."""
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
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
        
    def run_clustering(self, X: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """Run clustering analysis."""
        self.clustering_model = FinancialClusteringModel(n_clusters)
        return self.clustering_model.fit_predict(X)
    
    def run_classification(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run classification model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.classification_model = RiskClassificationModel()
        self.classification_model.fit(X_train, y_train)
        
        train_metrics = self.classification_model.evaluate(X_train, y_train)
        test_metrics = self.classification_model.evaluate(X_test, y_test)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    def run_forecasting(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run savings forecasting."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.forecast_model = SavingsForecastModel()
        self.forecast_model.fit(X_train, y_train)
        
        train_metrics = self.forecast_model.evaluate(X_train, y_train)
        test_metrics = self.forecast_model.evaluate(X_test, y_test)
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    def run_anomaly_detection(self, X: np.ndarray) -> np.ndarray:
        """Run anomaly detection."""
        self.anomaly_model = AnomalyDetectionModel(contamination=0.02)
        return self.anomaly_model.fit_predict(X)


def train_all_models(users_df: pd.DataFrame, monthly_df: pd.DataFrame) -> dict:
    """Train all models and return results."""
    
    from preprocessing import (prepare_clustering_data, prepare_classification_data,
                              prepare_regression_data)
    from sklearn.preprocessing import StandardScaler
    
    results = {}
    
    print("=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)
    
    # Preprocess users to create derived features
    from preprocessing import FinancialDataPreprocessor
    preprocessor = FinancialDataPreprocessor()
    users_df = preprocessor.preprocess_users(users_df)
    monthly_df = preprocessor.preprocess_monthly(monthly_df)
    
    # 1. Clustering
    print("\n[1/4] Training K-Means Clustering...")
    clustering_data = prepare_clustering_data(users_df)
    pipeline = FinancialMLPipeline()
    clusters = pipeline.run_clustering(clustering_data.values, n_clusters=5)
    users_df['cluster'] = clusters
    
    print(f"   âœ“ Clustered {len(users_df)} users into 5 segments")
    silhouette = pipeline.clustering_model.get_silhouette_score(clustering_data.values)
    print(f"   âœ“ Silhouette Score: {silhouette:.4f}")
    results['silhouette'] = silhouette
    
    # 2. Classification
    print("\n[2/4] Training Random Forest Classifier...")
    X_class, y_class, label_encoder = prepare_classification_data(users_df)
    results['classification'] = pipeline.run_classification(X_class.values, y_class)
    
    print(f"   âœ“ Test Accuracy: {results['classification']['test_metrics']['accuracy']:.4f}")
    print(f"   âœ“ CV Score: {results['classification']['test_metrics']['cv_mean']:.4f} Â+/- {results['classification']['test_metrics']['cv_std']:.4f}")
    
    # 3. Forecasting
    print("\n[3/4] Training Gradient Boosting Regressor...")
    X_reg, y_reg = prepare_regression_data(monthly_df)
    results['forecasting'] = pipeline.run_forecasting(X_reg.values, y_reg)
    
    print(f"   âœ“ Test RÂ2 Score: {results['forecasting']['test_metrics']['r2']:.4f}")
    print(f"   âœ“ Test RMSE: ${results['forecasting']['test_metrics']['rmse']:.2f}")
    
    # 4. Anomaly Detection
    print("\n[4/4] Training Isolation Forest...")
    spending_cols = ['Housing', 'Transportation', 'Food', 'Healthcare', 
                    'Entertainment', 'Shopping', 'Education', 'Subscriptions',
                    'Insurance', 'Miscellaneous']
    anomaly_data = monthly_df[spending_cols].fillna(0)
    anomalies = pipeline.run_anomaly_detection(anomaly_data.values)
    monthly_df['ml_anomaly'] = (anomalies == -1).astype(int)
    
    anomaly_count = (anomalies == -1).sum()
    print(f"   - Detected {anomaly_count} anomalies ({anomaly_count/len(monthly_df)*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Test model training
    from data_generator import generate_full_dataset
    
    users_df, monthly_df = generate_full_dataset(1000)
    results = train_all_models(users_df, monthly_df)
