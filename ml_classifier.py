import os
import logging
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any

# Path to the pre-trained model pipeline (scaler + XGBClassifier)
# Can be overridden via the ML_MODEL_PATH environment variable
DEFAULT_MODEL_PATH = "/app/models/xgb_classifier.pipeline.joblib"
MODEL_PATH = os.getenv("ML_MODEL_PATH", DEFAULT_MODEL_PATH)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}'
    ))
    logger.addHandler(ch)


class MLClassifier:
    """
    ML-based movement classifier using a pre-trained XGBoost pipeline.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Load the pipeline from disk. Raises FileNotFoundError or AttributeError
        with clear, logged error messages if something is misconfigured.
        """
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            logger.error(f"ML model file not found at: {self.model_path}")
            raise FileNotFoundError(f"Please ensure the model exists at {self.model_path}")

        logger.info(f"Loading ML pipeline from {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        logger.info("ML pipeline loaded successfully.")

        # Feature names should have been attached at training time
        self.feature_names = getattr(self.pipeline, "feature_names", None)
        if self.feature_names is None:
            logger.error("Loaded pipeline missing `feature_names` attribute")
            raise AttributeError("Loaded pipeline missing `feature_names` attribute")

    def classify(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param data: dict of raw features (keys must match feature_names)
        :return: {"movement_type": <label>, "expected_move_pct": <prob*100>}
        """
        # Build a one‐row DataFrame in the correct column order
        try:
            df = pd.DataFrame([data], columns=self.feature_names)
        except Exception as e:
            logger.error(f"Error constructing input DataFrame: {e!r}")
            raise

        # Predict class probabilities
        proba = self.pipeline.predict_proba(df)[0]
        # Retrieve class labels from the XGBClassifier step (named "xgb" in the pipeline)
        try:
            classes = self.pipeline.named_steps["xgb"].classes_
        except (KeyError, AttributeError) as e:
            logger.error(f"Failed to retrieve classes from pipeline: {e!r}")
            raise

        # Select the highest‐probability class
        idx = int(np.argmax(proba))
        movement_type = classes[idx]
        expected_move_pct = float(proba[idx] * 100.0)

        logger.debug(
            f"Classified movement_type={movement_type}, expected_move_pct={expected_move_pct:.2f}"
        )

        return {
            "movement_type": movement_type,
            "expected_move_pct": expected_move_pct
        }