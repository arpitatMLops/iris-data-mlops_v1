# src/model_training.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

FEATURES = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']

def model_train(input_path: str, model_path: str, n_estimators: int = 20):
    print(f"Training with input={input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Training data not found: {input_path}")

    df = pd.read_csv(input_path)
    if not set(FEATURES).issubset(df.columns) or 'target' not in df.columns:
        raise ValueError("Input must contain iris feature columns and 'target'")

    X = df[FEATURES]
    y = df['target']

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"accuracy={acc:.4f}")

    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_path, "model.joblib"))
    print(f"[INFO] Saved model to {model_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    # IMPORTANT: SageMaker Training mounts your channel here:
    # /opt/ml/input/data/<channel-name>/
    parser.add_argument("--input-path", type=str,
                        default="/opt/ml/input/data/train/processed.csv")
    # IMPORTANT: SageMaker picks up models from this folder
    parser.add_argument("--model-path", type=str, default="/opt/ml/model")
    parser.add_argument("--n-estimators", type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    a = parse_args()
    model_train(a.input_path, a.model_path, n_estimators=a.n_estimators)
