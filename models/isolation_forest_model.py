# models/isolation_forest_model.py
"""
Isolation Forest Detector — Phase 2 Individual Model.

Unsupervised anomaly detection. Treats any non-zero class as anomaly.
Maps IF output (-1=anomaly, 1=normal) to our 3-class system:
  anomaly predicted  →  label is mapped using a threshold on anomaly score
  normal  predicted  →  0 (Normal)

For evaluation, IF is treated as binary (normal vs attack) then upcast
to 3-class (attack = class 2 Business Logic, as IF cannot distinguish attack types).
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "saved_models", "if_model.pkl"
)


class IsolationForestDetector:
    """
    Isolation Forest wrapper for unsupervised anomaly detection.

    Parameters
    ----------
    contamination  : expected fraction of anomalies (default 'auto')
    n_estimators   : number of isolation trees
    random_state   : reproducibility seed
    """

    def __init__(
        self,
        contamination: float = 0.15,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.model         = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None) -> "IsolationForestDetector":
        """
        Train Isolation Forest (unsupervised — y_train is ignored but accepted
        for API compatibility with the training pipeline).
        """
        print(f"[IF] Training IsolationForest ({self.n_estimators} trees, "
              f"contamination={self.contamination}) ...")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0,
        )
        self.model.fit(X_train)
        print("[IF] Training complete.")
        self._save()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return 3-class labels.
        IF returns -1 (anomaly) or +1 (normal).
        We map: +1 → 0 (Normal), -1 → 2 (BusinessLogic attack proxy).
        """
        self._ensure_loaded()
        raw = self.model.predict(X)
        # -1 → 2 (anomaly = attack), 1 → 0 (normal)
        return np.where(raw == 1, 0, 2).astype(np.int32)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores (lower = more anomalous).
        Useful as input feature for hybrid IF+ANN model.
        """
        self._ensure_loaded()
        return self.model.score_samples(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Pseudo-probability via normalised anomaly scores.
        Returns [N, 3] array. Classes: [Normal, CredStuffing, BizLogic].
        """
        self._ensure_loaded()
        scores = self.anomaly_scores(X)
        # Normalise scores to [0, 1]: higher score = more normal
        min_s, max_s = scores.min(), scores.max()
        norm = (scores - min_s) / (max_s - min_s + 1e-9)
        # p_normal = norm, p_attack (class 2) = 1 - norm, class 1 = 0
        proba = np.zeros((len(X), 3), dtype=np.float32)
        proba[:, 0] = norm
        proba[:, 2] = 1.0 - norm
        return proba

    def _save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"[IF] Model saved → {MODEL_PATH}")

    def load(self) -> "IsolationForestDetector":
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"IF model not found at {MODEL_PATH}. Run training first.")
        self.model = joblib.load(MODEL_PATH)
        print(f"[IF] Model loaded from {MODEL_PATH}")
        return self

    def _ensure_loaded(self):
        if self.model is None:
            self.load()
