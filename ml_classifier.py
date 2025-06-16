import os
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any

# Path to the pre-trained model pipeline (scaler + XGBClassifier)
MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/xgb_classifier.pipeline.joblib")

class MLClassifier:
    """
    ML-based movement classifier using a pre-trained XGBoost pipeline.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        # The pipeline encapsulates preprocessing (StandardScaler) + XGBClassifier
        self.pipeline = joblib.load(model_path)
        # Feature names were attached at training time
        self.feature_names = getattr(self.pipeline, "feature_names", None)
        if self.feature_names is None:
            raise AttributeError("Loaded pipeline missing `feature_names` attribute")

    def classify(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param data: dict of raw features (keys must match feature_names)
        :return: {"movement_type": <label>, "expected_move_pct": <prob*100>}
        """
        # Build a one‐row DataFrame
        df = pd.DataFrame([data], columns=self.feature_names)
        # Predict class probabilities
        proba = self.pipeline.predict_proba(df)[0]
        classes = self.pipeline.named_steps["xgb"].classes_
        # Choose the highest‐probability class
        idx = int(np.argmax(proba))
        movement_type = classes[idx]
        expected_move_pct = float(proba[idx] * 100.0)
        return {
            "movement_type": movement_type,
            "expected_move_pct": expected_move_pct
        }
