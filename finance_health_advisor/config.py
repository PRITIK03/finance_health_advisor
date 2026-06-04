"""
Configuration Management for Finance Health Advisor
Centralizes all configurable parameters for models, data generation, and UI.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class DataConfig:
    """Data generation configuration."""
    total_users: int = 10_000
    months_per_user: int = 12
    anomaly_rate: float = 0.015
    seed: int = 42
    
    # Income distributions by employment type
    income_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Full-time': {'mean': 10.5, 'sigma': 0.4},
        'Part-time': {'mean': 9.5, 'sigma': 0.4},
        'Self-employed': {'mean': 10.8, 'sigma': 0.6},
        'Retired': {'mean': 9.8, 'sigma': 0.3},
        'Unemployed': {'mean': 9.0, 'sigma': 0.3},
    })
    
    # Expense ratio bounds
    expense_ratio_beta: tuple = (4, 2)  # Beta distribution params
    expense_ratio_range: tuple = (0.5, 0.9)
    
    # Spending category bounds
    spending_bounds: Dict[str, tuple] = field(default_factory=lambda: {
        'Housing': (0.25, 0.40),
        'Food': (0.10, 0.20),
        'Transportation': (0.08, 0.15),
        'Healthcare': (0.03, 0.10),
        'Entertainment': (0.05, 0.12),
        'Shopping': (0.03, 0.10),
        'Education': (0.02, 0.08),
        'Subscriptions': (0.02, 0.05),
        'Insurance': (0.03, 0.08),
    })


@dataclass
class ModelConfig:
    """ML model configuration."""
    # Clustering
    n_clusters: int = 5
    kmeans_n_init: int = 10
    kmeans_random_state: int = 42
    
    # Classification
    rf_n_estimators: int = 100
    rf_max_depth: int = 15
    rf_min_samples_split: int = 10
    rf_random_state: int = 42
    
    # Regression
    gb_n_estimators: int = 100
    gb_max_depth: int = 5
    gb_learning_rate: float = 0.1
    gb_random_state: int = 42
    
    # Anomaly Detection
    if_contamination: float = 0.02
    if_n_estimators: int = 100
    if_random_state: int = 42
    
    # KNN
    knn_n_neighbors: int = 10
    
    # Train/test split
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class UIConfig:
    """UI/Streamlit configuration."""
    page_title: str = "Finance Health Advisor"
    page_icon: str = "💰"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Pagination
    page_size_options: List[int] = field(default_factory=lambda: [50, 100, 200, 500])
    default_page_size: int = 100
    
    # Theme
    default_dark_mode: bool = False
    
    # Colors
    primary_color: str = "#6366f1"  # Indigo
    success_color: str = "#10b981"
    warning_color: str = "#f59e0b"
    danger_color: str = "#ef4444"
    
    # Chart settings
    chart_template: str = "plotly_white"
    chart_height: int = 400
    chart_width: Optional[int] = None


@dataclass
class FinancialConfig:
    """Financial calculation constants."""
    # FIRE
    safe_withdrawal_rate: float = 0.04
    inflation_rate: float = 0.03
    expected_nominal_return: float = 0.07
    
    # Emergency fund
    default_emergency_months: int = 6
    
    # Debt
    max_debt_simulation_months: int = 600
    
    # Stress test
    stress_test_months: int = 24
    
    # Health score weights
    health_score_weights: Dict[str, float] = field(default_factory=lambda: {
        'savings_rate': 0.30,
        'expense_ratio': 0.25,
        'credit_score': 0.15,
        'investment_ratio': 0.15,
        'debt_ratio': 0.15,
    })
    
    # Risk thresholds
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'Very Low': 80,
        'Low': 60,
        'Medium': 40,
        'High': 20,
        'Very High': 0,
    })


@dataclass
class AppConfig:
    """Main application configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    financial: FinancialConfig = field(default_factory=FinancialConfig)
    
    # Environment
    debug: bool = False
    log_level: str = "INFO"
    
    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = ".cache"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Data
        config.data.total_users = int(os.getenv("FHA_TOTAL_USERS", config.data.total_users))
        config.data.anomaly_rate = float(os.getenv("FHA_ANOMALY_RATE", config.data.anomaly_rate))
        config.data.seed = int(os.getenv("FHA_SEED", config.data.seed))
        
        # Models
        config.models.n_clusters = int(os.getenv("FHA_N_CLUSTERS", config.models.n_clusters))
        config.models.rf_n_estimators = int(os.getenv("FHA_RF_ESTIMATORS", config.models.rf_n_estimators))
        config.models.if_contamination = float(os.getenv("FHA_IF_CONTAMINATION", config.models.if_contamination))
        
        # UI
        config.ui.default_dark_mode = os.getenv("FHA_DARK_MODE", "false").lower() == "true"
        config.ui.default_page_size = int(os.getenv("FHA_PAGE_SIZE", config.ui.default_page_size))
        
        # Debug
        config.debug = os.getenv("FHA_DEBUG", "false").lower() == "true"
        config.log_level = os.getenv("FHA_LOG_LEVEL", config.log_level)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        
        if self.data.total_users < 100:
            warnings.append("Total users < 100 may produce unreliable ML results")
        
        if not 0 < self.data.anomaly_rate < 0.5:
            warnings.append("Anomaly rate should be between 0 and 0.5")
        
        if self.models.n_clusters < 2:
            warnings.append("n_clusters must be >= 2")
        
        if not 0 < self.models.test_size < 1:
            warnings.append("test_size must be between 0 and 1")
        
        if not 0 < self.models.if_contamination < 0.5:
            warnings.append("Isolation Forest contamination must be between 0 and 0.5")
        
        weight_sum = sum(self.financial.health_score_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            warnings.append(f"Health score weights sum to {weight_sum:.2f}, expected 1.0")
        
        return warnings


# Global configuration instance
CONFIG = AppConfig.from_env()

# Validate on import
_CONFIG_WARNINGS = CONFIG.validate()
if _CONFIG_WARNINGS and CONFIG.debug:
    import warnings as py_warnings
    for w in _CONFIG_WARNINGS:
        py_warnings.warn(w, UserWarning)