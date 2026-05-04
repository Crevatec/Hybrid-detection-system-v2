# models/hybrid_if_ann.py
"""
Hybrid IF + ANN Detector — Phase 3 Ensemble Model.

Strategy: Anomaly Score Augmentation
  1. Isolation Forest generates anomaly scores for every sample.
  2. These scores are appended as an extra feature to the input.
  3. ANN is trained on [original features + IF anomaly score].

This gives the ANN an unsupervised "anomaly signal" that helps it
distinguish subtle deviations without relying solely on labelled data.
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_class_weight
import joblib

IF_PATH  = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "if_ann_if.pkl")
ANN_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "if_ann_ann.keras")


class HybridIFANN:
    """
    Isolation Forest anomaly-score-augmented ANN.

    Parameters
    ----------
    if_contamination : expected anomaly fraction
    ann_epochs       : epochs for the ANN
    ann_batch_size   : batch size for the ANN
    """

    def __init__(
        self,
        if_contamination: float = 0.15,
        if_n_estimators: int = 150,
        ann_epochs: int = 25,
        ann_batch_size: int = 1024,
        random_state: int = 42,
    ):
        self.if_contamination = if_contamination
        self.if_n_estimators  = if_n_estimators
        self.ann_epochs       = ann_epochs
        self.ann_batch_size   = ann_batch_size
        self.random_state     = random_state
        self.iso_forest       = None
        self.ann              = None
        self._score_min       = None
        self._score_max       = None

    def _normalise_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalise raw IF scores to [0, 1]."""
        norm = (scores - self._score_min) / (self._score_max - self._score_min + 1e-9)
        return norm.reshape(-1, 1).astype(np.float32)

    def _build_ann(self, input_dim: int) -> tf.keras.Model:
        inp = layers.Input(shape=(input_dim,), name="augmented_features")
        x = layers.BatchNormalization()(inp)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(3, activation="softmax", name="predictions")(x)
        model = models.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HybridIFANN":
        """Train IF then ANN on anomaly-score-augmented features."""
        print("[IF+ANN] Step 1/2 — Training Isolation Forest ...")
        self.iso_forest = IsolationForest(
            n_estimators=self.if_n_estimators,
            contamination=self.if_contamination,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.iso_forest.fit(X_train)
        os.makedirs(os.path.dirname(IF_PATH), exist_ok=True)
        joblib.dump(self.iso_forest, IF_PATH)

        raw_scores = self.iso_forest.score_samples(X_train)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        norm_scores = self._normalise_scores(raw_scores)

        print("[IF+ANN] Step 2/2 — Training ANN on anomaly-augmented features ...")
        X_augmented = np.hstack([X_train, norm_scores])   # [N, features + 1]

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cwd = dict(zip(classes.tolist(), weights.tolist()))

        self.ann = self._build_ann(X_augmented.shape[1])
        cb_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        self.ann.fit(
            X_augmented, y_train,
            epochs=self.ann_epochs,
            batch_size=self.ann_batch_size,
            validation_split=0.1,
            class_weight=cwd,
            callbacks=cb_list,
            verbose=1,
        )
        self.ann.save(ANN_PATH)
        # Save normalisation constants alongside the IF model
        import json
        meta = {"score_min": self._score_min, "score_max": self._score_max}
        with open(IF_PATH.replace(".pkl", "_meta.json"), "w") as f:
            json.dump(meta, f)
        print("[IF+ANN] Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        raw_scores  = self.iso_forest.score_samples(X)
        norm_scores = self._normalise_scores(raw_scores)
        X_aug       = np.hstack([X, norm_scores])
        proba       = self.ann.predict(X_aug, verbose=0)
        return np.argmax(proba, axis=1).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        raw_scores  = self.iso_forest.score_samples(X)
        norm_scores = self._normalise_scores(raw_scores)
        X_aug       = np.hstack([X, norm_scores])
        return self.ann.predict(X_aug, verbose=0).astype(np.float32)

    def load(self) -> "HybridIFANN":
        import json
        self.iso_forest = joblib.load(IF_PATH)
        self.ann        = tf.keras.models.load_model(ANN_PATH)
        meta_path = IF_PATH.replace(".pkl", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self._score_min = meta["score_min"]
            self._score_max = meta["score_max"]
        print("[IF+ANN] Models loaded.")
        return self

    def _ensure_loaded(self):
        if self.iso_forest is None or self.ann is None:
            self.load()
