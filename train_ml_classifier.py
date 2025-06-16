import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Path to your labeled training CSV; must include all features + "movement_type" column
TRAIN_DATA_PATH = "data/movement_training_data.csv"

# Where to save the trained pipeline
MODEL_DIR      = "models"
MODEL_FILENAME = "xgb_classifier.pipeline.joblib"

def train():
    # 1) Load dataset
    df = pd.read_csv(TRAIN_DATA_PATH)
    X = df.drop(columns=["movement_type"])
    y = df["movement_type"]

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 3) Build pipeline: scaling + XGBoost
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            use_label_encoder=False,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        ))
    ])

    # 4) Train
    pipeline.fit(X_train, y_train)

    # 5) Validate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc:.4f}")

    # 6) Save feature names on the pipeline for inference
    pipeline.feature_names = X.columns.tolist()

    # 7) Persist pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(pipeline, model_path)
    print(f"✅ Trained pipeline saved to {model_path}")

if __name__ == "__main__":
    train()
