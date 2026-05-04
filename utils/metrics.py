# utils/metrics.py
"""
Evaluation utilities for Hybrid Behavioral Detection System.
Computes Accuracy, Precision, Recall, F1-Score (macro & weighted).
Saves/loads metrics as JSON for comparison charts.
"""

import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    verbose: bool = True,
) -> dict:
    """
    Compute and return a metrics dict for a model's predictions.

    Returns
    -------
    dict with keys: model_name, accuracy, precision, recall, f1, report
    """
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    f1_macro  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "CredentialStuffing", "BusinessLogic"],
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "model_name": model_name,
        "accuracy":   round(float(accuracy),  4),
        "precision":  round(float(precision), 4),
        "recall":     round(float(recall),    4),
        "f1":         round(float(f1),        4),
        "f1_macro":   round(float(f1_macro),  4),
        "confusion_matrix": cm,
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Model: {model_name}")
        print(f"{'='*55}")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 (w)   : {f1:.4f}")
        print(f"  F1 (mac) : {f1_macro:.4f}")
        print(f"\n{report}")

    return metrics


def save_metrics(metrics: dict, filename: str = None) -> str:
    """Save metrics dict to JSON in the outputs/metrics directory."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    if filename is None:
        safe_name = metrics["model_name"].replace(" ", "_").replace("+", "plus")
        filename = f"{safe_name}.json"
    path = os.path.join(METRICS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved → {path}")
    return path


def load_all_metrics() -> list:
    """Load all saved metrics JSON files and return a list of dicts."""
    if not os.path.exists(METRICS_DIR):
        return []
    results = []
    for fname in sorted(os.listdir(METRICS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(METRICS_DIR, fname)) as f:
                results.append(json.load(f))
    return results
