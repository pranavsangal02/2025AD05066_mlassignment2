"""
Shared utilities for the five model training scripts.

Every model script imports from this module so the dataset, the train/test
split and the preprocessing pipeline are identical across all of them. That is
what makes the comparison table a fair comparison.

Run any model script from the repository root, for example:
    python model/logistic_regression.py
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "Revenue"

# Paths resolve relative to the repository root (the parent of model/),
# so the scripts work no matter which directory you launch them from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "online_shoppers_intention_2.csv"
MODEL_DIR = REPO_ROOT / "model"

NUMERIC = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]


def load_split():
    """Load the dataset and return a stratified 80/20 train/test split."""
    df = pd.read_csv(DATA_PATH)
    df["Weekend"] = df["Weekend"].astype(bool)
    df["Revenue"] = df["Revenue"].astype(bool)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)          # True -> 1, False -> 0
    return train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def build_preprocessor(dense=False):
    """Scale numeric columns and one-hot encode categorical columns.

    dense=True returns a dense matrix, required by GaussianNB.
    Fitting happens inside the Pipeline on the training split only, so no
    test-set statistics leak into training.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
         CATEGORICAL),
    ])


def evaluate(pipe, X_test, y_test):
    """Return the six assignment metrics for a fitted pipeline."""
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]     # probabilities for AUC
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def report(name, pipe, X_test, y_test):
    """Print metrics and the confusion matrix for one model."""
    metrics = evaluate(pipe, X_test, y_test)
    print(f"\n{name}")
    print("-" * len(name))
    for key, value in metrics.items():
        print(f"  {key:<10} {value:.4f}")
    print("  Confusion matrix [[TN FP] [FN TP]]:")
    print("   ", confusion_matrix(y_test, pipe.predict(X_test)).tolist())
    return metrics


def save(pipe, filename):
    """Persist a fitted pipeline to model/<filename>."""
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump(pipe, path, compress=3)
    print(f"  saved -> {path.relative_to(REPO_ROOT)}")
