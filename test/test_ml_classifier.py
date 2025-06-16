# tests/test_ml_classifier.py

import os
import pytest
import pandas as pd
import numpy as np
import joblib
from ml_classifier import MLClassifier

MODEL_PATH = "models/xgb_classifier.pipeline.joblib"

@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists(tmp_path_factory):
    # If you don't have a real model, write a dummy pipeline
    if not os.path.exists(MODEL_PATH):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        # dummy pipeline with fixed feature names
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", RandomForestClassifier())
        ])
        pipe.feature_names = ["delta","gamma","rsi"]  # minimal
        joblib.dump(pipe, MODEL_PATH)
    yield

def test_ml_classifier_loads():
    clf = MLClassifier(model_path=MODEL_PATH)
    assert hasattr(clf, "pipeline")
    assert isinstance(clf.feature_names, list)

def test_ml_classifier_classify():
    clf = MLClassifier(model_path=MODEL_PATH)
    sample = {fn: 0.0 for fn in clf.feature_names}
    out = clf.classify(sample)
    assert "movement_type" in out
    assert "expected_move_pct" in out
    assert isinstance(out["movement_type"], str)
    assert isinstance(out["expected_move_pct"], float)
