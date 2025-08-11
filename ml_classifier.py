"""
Wrapper around a pre-trained pipeline for movement classification.

Loads an sklearn pipeline (with a final XGBoost classifier) from disk,
optionally downloading from S3 if missing. Uses the pipeline's RAW input
schema (feature_names_in_) and safely fills any missing inputs with
defaults so prediction doesn't crash if a feature is absent.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import joblib
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import numpy as np
import pandas as pd

# Local model path (inside the container)
DEFAULT_MODEL_PATH = "/app/models/xgb_classifier.pipeline.joblib"
MODEL_PATH = os.getenv("ML_MODEL_PATH", DEFAULT_MODEL_PATH)

# S3 settings for auto-download if file is missing
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "models/xgb_classifier.pipeline.joblib")

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":"%(message)s"}'
    ))
    logger.addHandler(_h)


def _download_from_s3(bucket: str, key: str, dest: str) -> None:
    """Download the model artifact from S3 to the local path."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s3 = boto3.client("s3")
    logger.info(f"Downloading model from s3://{bucket}/{key}")
    s3.download_file(bucket, key, dest)
    logger.info("Model downloaded successfully.")


class MLClassifier:
    """
    ML-based movement classifier using a pre-trained sklearn Pipeline.
    It relies on `pipeline.feature_names_in_` for the RAW input columns.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.model_path = model_path

        # Ensure model exists (download if configured)
        if not os.path.exists(self.model_path):
            if S3_BUCKET:
                try:
                    _download_from_s3(S3_BUCKET, S3_MODEL_KEY, self.model_path)
                except (ClientError, BotoCoreError) as exc:
                    raise FileNotFoundError(f"Could not obtain model from S3: {exc}") from exc
            else:
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info(f"Loading ML pipeline from {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        logger.info("ML pipeline loaded successfully.")

        # RAW input schema expected by the pipeline's preprocessor
        cols = getattr(self.pipeline, "feature_names_in_", None)
        if cols is None or len(cols) == 0:
            raise AttributeError("Pipeline missing feature_names_in_. Re-export the model with sklearn>=1.0.")
        self.input_cols: List[str] = list(cols)

        # NOTE: Some artifacts also store transformed feature names in `feature_names`.
        # Those are post-encoding and NOT what we should pass as inputs. We keep them
        # only for reference.
        self.transformed_cols: List[str] = list(getattr(self.pipeline, "feature_names", []) or [])

    # ---- Defaults for any missing inputs -------------------------------------
    def _default_for(self, col: str) -> Any:
        if col == "time_of_day":
            # Categorical bucket; orchestrator typically provides a string like "MORNING"
            return "OFF_HOURS"
        # numeric default
        return 0.0

    def classify(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single sample and return movement type and expected move percentage.

        Parameters
        ----------
        data : dict
            Raw features; keys SHOULD match pipeline.feature_names_in_. Any absent
            keys are filled with safe defaults.

        Returns
        -------
        dict: {"movement_type": <label>, "expected_move_pct": <prob*100>}
        """
        # Build row aligned to RAW input schema
        row = {}
        missing = []
        for col in self.input_cols:
            if col in data and data[col] is not None:
                row[col] = data[col]
            else:
                row[col] = self._default_for(col)
                missing.append(col)

        if missing:
            logger.info(f"Filled missing model inputs with defaults: {missing}")

        df = pd.DataFrame([row], columns=self.input_cols)

        proba = self.pipeline.predict_proba(df)[0]
        try:
            classes = self.pipeline.named_steps["xgb"].classes_
        except Exception:
            # Some pipelines store classes_ on the overall pipeline
            classes = getattr(self.pipeline, "classes_", None)
            if classes is None:
                raise AttributeError("Could not locate classifier classes_ in the pipeline.")

        idx = int(np.argmax(proba))
        return {
            "movement_type": classes[idx],
            "expected_move_pct": float(proba[idx] * 100.0),
        }
