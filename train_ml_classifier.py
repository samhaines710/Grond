#!/usr/bin/env python3
import os
import argparse
import logging
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def setup_logging():
    """
    Configure root logger for structured output.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}'
    ))
    root.handlers.clear()
    root.addHandler(handler)

def train(
    train_csv: str,
    label_col: str,
    model_dir: str,
    model_filename: str,
    test_size: float,
    random_state: int
):
    """
    1) Load CSV
    2) Split train/test
    3) Build & fit Pipeline
    4) Validate on test set
    5) Attach feature_names + dump to disk
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading training data from {train_csv}")
    df = pd.read_csv(train_csv)

    if label_col not in df.columns:
        logger.error(f"Label column '{label_col}' not found in {train_csv}")
        raise ValueError(f"Missing label column: {label_col}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    logger.info("Building pipeline (StandardScaler + XGBClassifier)")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            use_label_encoder=False,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state
        ))
    ])

    logger.info("Fitting pipeline on training data")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating on test set")
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Validation Accuracy: {acc:.4f}")

    # Attach feature names for inference alignment
    pipeline.feature_names = X.columns.tolist()
    logger.info(f"Attached feature_names ({len(pipeline.feature_names)} features) to pipeline")

    # Persist pipeline
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(pipeline, model_path)
    logger.info(f"✅ Trained pipeline saved to {model_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and persist the XGBClassifier movement‐type pipeline."
    )
    parser.add_argument(
        "--train-csv",
        default=os.getenv("TRAIN_DATA_PATH", "data/movement_training_data.csv"),
        help="Path to labeled training CSV (must include '<label_col>' column)."
    )
    parser.add_argument(
        "--label-col",
        default=os.getenv("LABEL_COL", "movement_type"),
        help="Name of the target column in the CSV."
    )
    parser.add_argument(
        "--model-dir",
        default=os.getenv("MODEL_DIR", "models"),
        help="Directory to write the trained pipeline."
    )
    parser.add_argument(
        "--model-filename",
        default=os.getenv("MODEL_FILENAME", "xgb_classifier.pipeline.joblib"),
        help="Filename for the saved pipeline."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=float(os.getenv("TEST_SIZE", "0.2")),
        help="Fraction of data to hold out for validation."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=int(os.getenv("RANDOM_STATE", "42")),
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    train(
        train_csv=args.train_csv,
        label_col=args.label_col,
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        test_size=args.test_size,
        random_state=args.random_state
    ) 