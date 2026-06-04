"""
Model Persistence Module for Finance Health Advisor
Handles saving and loading of trained ML models using joblib.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from logging_config import get_logger

logger = get_logger(__name__)


class ModelPersistence:
    """Handles saving and loading of ML models and pipeline components."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_pipeline(self, pipeline, filename: str = "fha_pipeline.joblib") -> str:
        """Save the entire ML pipeline."""
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(pipeline, filepath)
        logger.info(f"Pipeline saved to {filepath}")
        return filepath
    
    def load_pipeline(self, filename: str = "fha_pipeline.joblib"):
        """Load the ML pipeline."""
        filepath = os.path.join(self.model_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline not found at {filepath}")
        pipeline = joblib.load(filepath)
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
    
    def save_model(self, model, filename: str) -> str:
        """Save a single model component."""
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filename: str):
        """Load a single model component."""
        filepath = os.path.join(self.model_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found at {filepath}")
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def save_preprocessor(self, preprocessor, filename: str = "preprocessor.joblib") -> str:
        """Save the data preprocessor."""
        return self.save_model(preprocessor, filename)
    
    def load_preprocessor(self, filename: str = "preprocessor.joblib"):
        """Load the data preprocessor."""
        return self.load_model(filename)
    
    def save_scaler(self, scaler, filename: str) -> str:
        """Save a fitted scaler."""
        return self.save_model(scaler, filename)
    
    def load_scaler(self, filename: str):
        """Load a fitted scaler."""
        return self.load_model(filename)
    
    def save_label_encoder(self, encoder, filename: str = "label_encoder.joblib") -> str:
        """Save a fitted label encoder."""
        return self.save_model(encoder, filename)
    
    def load_label_encoder(self, filename: str = "label_encoder.joblib"):
        """Load a fitted label encoder."""
        return self.load_model(filename)
    
    def list_saved_models(self) -> list:
        """List all saved model files."""
        if not os.path.exists(self.model_dir):
            return []
        return [f for f in os.listdir(self.model_dir) if f.endswith('.joblib')]


def save_training_artifacts(pipeline, preprocessor, label_encoder, results: dict, 
                           model_dir: str = "models") -> Dict[str, str]:
    """Save all training artifacts in one call."""
    persistence = ModelPersistence(model_dir)
    saved = {}
    
    saved['pipeline'] = persistence.save_pipeline(pipeline)
    saved['preprocessor'] = persistence.save_preprocessor(preprocessor)
    if label_encoder:
        saved['label_encoder'] = persistence.save_label_encoder(label_encoder)
    
    # Save results summary
    results_path = os.path.join(model_dir, "training_results.json")
    import json
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    saved['results'] = results_path
    
    logger.info(f"All training artifacts saved to {model_dir}")
    return saved


def load_training_artifacts(model_dir: str = "models") -> Tuple[Any, Any, Any, dict]:
    """Load all training artifacts in one call."""
    persistence = ModelPersistence(model_dir)
    
    pipeline = persistence.load_pipeline()
    preprocessor = persistence.load_preprocessor()
    label_encoder = persistence.load_label_encoder()
    
    results_path = os.path.join(model_dir, "training_results.json")
    with open(results_path, 'r') as f:
        import json
        results = json.load(f)
    
    logger.info(f"All training artifacts loaded from {model_dir}")
    return pipeline, preprocessor, label_encoder, results