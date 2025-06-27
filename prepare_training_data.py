#!/usr/bin/env python3
import os
import argparse
import logging

import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":"%(message)s"}'
)
logger = logging.getLogger(__name__)

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TRAIN_CSV   = "data/movement_training_data.csv"
DEFAULT_MODEL_DIR   = "models"
DEFAULT_MODEL_FILE  = "xgb_classifier.pipeline.joblib"

# ─── Arg Parsing ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train XGB classifier on your movement data"
    )
    p.add_argument(
        "--train-csv", type=str, default=DEFAULT_TRAIN_CSV,
        help="path to your CSV with features + movement_type"
    )
    p.add_argument(
        "--label-col", type=str, required=True,
        help="the name of your target column (e.g. movement_type)"
    )
    p.add_argument(
        "--model-dir", type=str, default=DEFAULT_MODEL_DIR,
        help="directory to save the trained pipeline"
    )
    p.add_argument(
        "--model-filename", type=str, default=DEFAULT_MODEL_FILE,
        help="filename (inside model-dir) for the pipeline"
    )
    return p.parse_args()

# ─── Training Function ─────────────────────────────────────────────────────────
def train(train_csv: str, label_col: str, model_dir: str, model_filename: str):
    # 1) Load data
    logger.info(f"Loading training data from {train_csv}")
    df = pd.read_csv(train_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in {train_csv}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # 2) Split
    logger.info("Splitting data: test_size=0.2, random_state=42")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 3) Identify numeric vs categorical
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # explicitly list any string/categorical columns here:
    cat_cols = ["time_of_day"]

    logger.info(f"Numeric columns: {num_cols}")
    logger.info(f"Categorical columns: {cat_cols}")

    # 4) Build ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse=False),
            cat_cols
        )
    ])

    # 5) Full pipeline: preprocess → XGB
    pipeline = Pipeline(steps=[
        ("pre", preprocessor),
        ("xgb", xgb.XGBClassifier(
            use_label_encoder=False,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        ))
    ])

    # 6) Train
    logger.info("Fitting pipeline on training data")
    pipeline.fit(X_train, y_train)

    # 7) Evaluate
    logger.info("Predicting on test set")
    preds = pipeline.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    logger.info(f"Validation Accuracy: {acc:.4f}")

    # 8) Save feature names (for downstream inference)
    cat_feature_names = (
        pipeline
        .named_steps["pre"]
        .named_transformers_["cat"]
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    feature_names = num_cols + cat_feature_names
    pipeline.feature_names = feature_names

    # 9) Persist pipeline
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(pipeline, model_path)
    logger.info(f"✅ Trained pipeline saved to {model_path}")

# ─── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(
        train_csv      = args.train_csv,
        label_col      = args.label_col,
        model_dir      = args.model_dir,
        model_filename = args.model_filename
    )
