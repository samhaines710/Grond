"""Wrapper around a pre-trained XGBoost pipeline for movement classification.

This class loads a pipeline from disk or optionally downloads it from S3.
It provides a `classify` method to return a movement type and expected move
percentage.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import joblib
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import numpy as np
import pandas as pd

# Where to look for the model locally (inside the container)
DEFAULT_MODEL_PATH = "/app/models/xgb_classifier.pipeline.joblib"
MODEL_PATH = os.getenv("ML_MODEL_PATH", DEFAULT_MODEL_PATH)

# S3 settings for auto-download if the file is missing
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "models/xgb_classifier.pipeline.joblib")

# Structured logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"module":"%(module)s","message":"%(message)s"}'
        )
    )
    logger.addHandler(handler)


def download_from_s3(bucket: str, key: str, dest: str) -> None:
    """
    Download the model artifact from S3 to the local path.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s3 = boto3.client("s3")
    try:
        logger.info(f"Downloading model from s3://{bucket}/{key}")
        s3.download_file(bucket, key, dest)
        logger.info("Model downloaded successfully.")
    except (ClientError, BotoCoreError) as exc:
        logger.error(f"Failed to download model from S3: {exc}")
        raise


class MLClassifier:
    """
    ML-based movement classifier using a pre-trained XGBoost pipeline.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.model_path: str = model_path

        if not os.path.exists(self.model_path):
            if S3_BUCKET:
                download_from_s3(S3_BUCKET, S3_MODEL_KEY, self.model_path)
            else:
                logger.error(f"Model file not found at: {self.model_path}")
                raise FileNotFoundError(
                    f"Please ensure the model exists at {self.model_path}"
                )

        logger.info(f"Loading ML pipeline from {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        logger.info("ML pipeline loaded successfully.")

        # Ensure feature_names were attached at training time
        self.feature_names = getattr(self.pipeline, "feature_names", None)
        if self.feature_names is None:
            logger.error("Loaded pipeline missing `feature_names` attribute")
            raise AttributeError(
                "Loaded pipeline missing `feature_names` attribute"
            )

    def classify(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single sample and return movement type and expected move percentage.

        Parameters
        ----------
        data : Dict[str, Any]
            dict of raw features; keys must match the pipeline's feature names.

        Returns
        -------
        Dict[str, Any]
            {"movement_type": movement_type, "expected_move_pct": percentage}
        """
        try:
            df = pd.DataFrame([data], columns=self.feature_names)
        except Exception as exc:
            logger.error(f"Error constructing input DataFrame: {exc!r}")
            raise

        proba = self.pipeline.predict_proba(df)[0]
        try:
            classes = self.pipeline.named_steps["xgb"].classes_
        except (KeyError, AttributeError) as exc:
            logger.error(f"Failed to retrieve classes from pipeline: {exc!r}")
            raise

        idx = int(np.argmax(proba))
        movement_type = classes[idx]
        expected_move_pct = float(proba[idx] * 100.0)

        logger.debug(
            "Classified movement_type=%s, expected_move_pct=%.2f",
            movement_type,
            expected_move_pct,
        )

        return {
            "movement_type": movement_type,
            "expected_move_pct": expected_move_pct,
        }
