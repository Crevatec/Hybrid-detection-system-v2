# train_hybrid.py
"""
Phase 3 — Train all four hybrid/ensemble models.

Run:
    python train_hybrid.py

Prerequisite: train_individual.py must have been run first
(scaler.pkl must exist in outputs/saved_models/).

Saves trained models to outputs/saved_models/
Saves evaluation metrics to outputs/metrics/
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_and_prepare_data
from utils.metrics import evaluate_model, save_metrics

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "api_security_dataset_2026.csv")
SAMPLE    = 300_000


def main():
    print("\n" + "="*60)
    print("  PHASE 3 — HYBRID ENSEMBLE MODEL TRAINING")
    print("="*60)

    df, X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(
        DATA_PATH, sample_size=SAMPLE, verbose=True
    )

    results = {}

    # ──────────────────────────────────────────────────────────────────────
    # 1. RF + ANN (Stacking)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/4] Hybrid: RF + ANN (Stacking)")
    from models.hybrid_rf_ann import HybridRFANN
    h1 = HybridRFANN(rf_n_estimators=150, ann_epochs=25)
    h1.fit(X_train, y_train)
    y_pred_h1 = h1.predict(X_test)
    m_h1 = evaluate_model(y_test, y_pred_h1, "RF + ANN")
    save_metrics(m_h1)
    results["RF + ANN"] = m_h1

    # ──────────────────────────────────────────────────────────────────────
    # 2. IF + ANN (Anomaly-Augmented)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[2/4] Hybrid: IF + ANN (Anomaly Score Augmentation)")
    from models.hybrid_if_ann import HybridIFANN
    h2 = HybridIFANN(if_contamination=0.15, ann_epochs=25)
    h2.fit(X_train, y_train)
    y_pred_h2 = h2.predict(X_test)
    m_h2 = evaluate_model(y_test, y_pred_h2, "IF + ANN")
    save_metrics(m_h2)
    results["IF + ANN"] = m_h2

    # ──────────────────────────────────────────────────────────────────────
    # 3. RF + LSTM (Parallel Weighted Averaging)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[3/4] Hybrid: RF + LSTM (Parallel Weighted Averaging)")
    from models.hybrid_rf_lstm import HybridRFLSTM
    h3 = HybridRFLSTM(rf_n_estimators=150, seq_len=5, lstm_epochs=25)
    h3.fit(X_train, y_train)
    y_pred_h3 = h3.predict(X_test)
    m_h3 = evaluate_model(y_test, y_pred_h3, "RF + LSTM")
    save_metrics(m_h3)
    results["RF + LSTM"] = m_h3

    # ──────────────────────────────────────────────────────────────────────
    # 4. ANN + LSTM + RF — Master Hybrid
    # ──────────────────────────────────────────────────────────────────────
    print("\n[4/4] Master Hybrid: ANN + LSTM + RF (Tri-model Soft Voting)")
    from models.hybrid_master import MasterHybrid
    h4 = MasterHybrid(rf_n_estimators=150, seq_len=5,
                      ann_epochs=25, lstm_epochs=20,
                      rf_weight=0.30, ann_weight=0.35, lstm_weight=0.35)
    h4.fit(X_train, y_train)
    y_pred_h4 = h4.predict(X_test)
    m_h4 = evaluate_model(y_test, y_pred_h4, "ANN + LSTM + RF (Master)")
    save_metrics(m_h4)
    results["ANN + LSTM + RF (Master)"] = m_h4

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  HYBRID MODELS — SUMMARY")
    print("="*60)
    print(f"{'Model':<28} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-"*60)
    for name, m in results.items():
        print(f"{name:<28} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f}")
    print("="*60)
    print("\nDone. Run evaluate_all.py to generate comparison charts.")


if __name__ == "__main__":
    main()
