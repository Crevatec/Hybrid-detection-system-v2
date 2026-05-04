# train_individual.py
"""
Phase 2 — Train all four individual models independently.

Run:
    python train_individual.py

Saves trained models to outputs/saved_models/
Saves evaluation metrics to outputs/metrics/
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from utils.data_loader import load_and_prepare_data
from utils.metrics import evaluate_model, save_metrics

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "api_security_dataset_2026.csv")
SAMPLE    = 300_000   # rows to load (adjust down if RAM is limited)


def main():
    print("\n" + "="*60)
    print("  PHASE 2 — INDIVIDUAL MODEL TRAINING")
    print("="*60)

    # ── Load & prepare data ────────────────────────────────────────────────
    df, X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(
        DATA_PATH, sample_size=SAMPLE, verbose=True
    )
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}\n")

    results = {}

    # ──────────────────────────────────────────────────────────────────────
    # 1. Random Forest
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/4] Random Forest")
    from models.random_forest_model import RandomForestDetector
    rf = RandomForestDetector(n_estimators=200)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    m_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")
    save_metrics(m_rf)
    results["Random Forest"] = m_rf

    # ──────────────────────────────────────────────────────────────────────
    # 2. Isolation Forest
    # ──────────────────────────────────────────────────────────────────────
    print("\n[2/4] Isolation Forest")
    from models.isolation_forest_model import IsolationForestDetector
    iso = IsolationForestDetector(contamination=0.15, n_estimators=200)
    iso.fit(X_train, y_train)
    y_pred_if = iso.predict(X_test)
    m_if = evaluate_model(y_test, y_pred_if, "Isolation Forest")
    save_metrics(m_if)
    results["Isolation Forest"] = m_if

    # ──────────────────────────────────────────────────────────────────────
    # 3. ANN
    # ──────────────────────────────────────────────────────────────────────
    print("\n[3/4] Artificial Neural Network (ANN)")
    from models.ann_model import ANNDetector
    ann = ANNDetector(epochs=30, batch_size=1024)
    ann.fit(X_train, y_train)
    y_pred_ann = ann.predict(X_test)
    m_ann = evaluate_model(y_test, y_pred_ann, "ANN")
    save_metrics(m_ann)
    results["ANN"] = m_ann

    # ──────────────────────────────────────────────────────────────────────
    # 4. LSTM
    # ──────────────────────────────────────────────────────────────────────
    print("\n[4/4] LSTM")
    from models.lstm_model import LSTMDetector
    lstm = LSTMDetector(seq_len=5, epochs=25, batch_size=512)
    lstm.fit(X_train, y_train)
    y_pred_lstm = lstm.predict(X_test)
    m_lstm = evaluate_model(y_test, y_pred_lstm, "LSTM")
    save_metrics(m_lstm)
    results["LSTM"] = m_lstm

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  INDIVIDUAL MODELS — SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-"*60)
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f}")
    print("="*60)
    print("\nDone. Run train_hybrid.py next.")


if __name__ == "__main__":
    main()
