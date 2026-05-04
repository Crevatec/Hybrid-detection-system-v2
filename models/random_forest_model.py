# models/random_forest_model.py
"""
Random Forest Detector — Phase 2 Individual Model.

Trains a supervised multi-class Random Forest on engineered features.
Saves model to outputs/saved_models/rf_model.pkl after training.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "saved_models", "rf_model.pkl"
)


class RandomForestDetector:
    """
    Random Forest Classifier wrapper for behavioral attack detection.

    Parameters
    ----------
    n_estimators : number of trees
    max_depth    : maximum tree depth (None = unlimited)
    random_state : reproducibility seed
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = None,
        random_state: int = 42,
    ):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.random_state  = random_state
        self.model         = None
        self.classes_      = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestDetector":
        """Train the Random Forest on labelled data."""
        print(f"[RF] Training RandomForest ({self.n_estimators} trees) ...")

        # Compute balanced class weights to handle imbalance
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=class_weight_dict,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1,
        )
        self.model.fit(X_train, y_train)
        self.classes_ = classes
        print("[RF] Training complete.")
        self._save()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        self._ensure_loaded()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates [N, 3]."""
        self._ensure_loaded()
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        """Return feature importance scores."""
        self._ensure_loaded()
        return self.model.feature_importances_

    def _save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"[RF] Model saved → {MODEL_PATH}")

    def load(self) -> "RandomForestDetector":
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"RF model not found at {MODEL_PATH}. Run training first.")
        self.model = joblib.load(MODEL_PATH)
        self.classes_ = self.model.classes_
        print(f"[RF] Model loaded from {MODEL_PATH}")
        return self

    def _ensure_loaded(self):
        if self.model is None:
            self.load()
