#!/usr/bin/env python3
"""
train_ml_classifier.py

Trains XGBClassifier on the movement data. Drops string columns so you
won’t hit “could not convert string to float” errors.
"""

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
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}'
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Train XGB classifier on movement data")
    p.add_argument("--train-csv",      default="data/movement_training_data.csv")
    p.add_argument("--label-col",      default="movement_type")
    p.add_argument("--model-dir",      default="models")
    p.add_argument("--model-filename", default="xgb_classifier.pipeline.joblib")
    return p.parse_args()

def train(train_csv, label_col, model_dir, model_filename):
    logger.info(f'"Loading training data from {train_csv}"')
    df = pd.read_csv(train_csv)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col!r}")

    # Drop all object (string) columns except the label itself
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    drop_cols = [c for c in obj_cols if c != label_col]
    X = df.drop(columns=drop_cols + [label_col])
    y = df[label_col]

    logger.info('"Splitting data (test_size=0.2, random_state=42)"')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Build preprocessing + model pipeline
    num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = []  # no categorical needed now

    logger.info(f'"Num cols={num_cols}, Cat cols={cat_cols}"')
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ),
    ], remainder="drop")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("xgb", xgb.XGBClassifier(
            use_label_encoder=False,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        ))
    ])

    logger.info('"Fitting pipeline on training data"')
    pipeline.fit(X_train, y_train)

    logger.info('"Validating…"')
    preds = pipeline.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    logger.info(f'"Validation Accuracy: {acc:.4f}"')

    # Save output feature names
    feature_names = (
        num_cols +
        pipeline.named_steps["pre"]
                .named_transformers_["cat"]
                .get_feature_names_out(cat_cols)
                .tolist()
    )
    pipeline.feature_names = feature_names

    # Persist pipeline
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, model_filename)
    joblib.dump(pipeline, out_path)
    logger.info(f'"✅ Saved pipeline to {out_path}"')

if __name__ == "__main__":
    args = parse_args()
    train(
        args.train_csv,
        args.label_col,
        args.model_dir,
        args.model_filename
    )
