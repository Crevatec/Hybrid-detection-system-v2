# models/hybrid_rf_ann.py
"""
Hybrid RF + ANN Detector — Phase 3 Ensemble Model.

Strategy: Stacking
  1. Random Forest is trained as base learner → outputs class probabilities [N, 3]
  2. ANN meta-learner is trained on [original features + RF probabilities]
     to learn to correct and combine signals.

This leverages RF's structured pattern recognition alongside ANN's
non-linear decision boundary refinement.
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
import joblib

from models.random_forest_model import RandomForestDetector

RF_PATH  = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "rf_ann_rf.pkl")
ANN_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "rf_ann_meta.keras")


class HybridRFANN:
    """
    RF + ANN Stacking Ensemble.

    Parameters
    ----------
    rf_n_estimators : trees in the base RF
    ann_epochs      : training epochs for meta-ANN
    ann_batch_size  : batch size for meta-ANN
    """

    def __init__(
        self,
        rf_n_estimators: int = 150,
        ann_epochs: int = 25,
        ann_batch_size: int = 1024,
        random_state: int = 42,
    ):
        self.rf_n_estimators = rf_n_estimators
        self.ann_epochs      = ann_epochs
        self.ann_batch_size  = ann_batch_size
        self.random_state    = random_state
        self.rf              = None
        self.ann             = None

    def _build_meta_ann(self, input_dim: int) -> tf.keras.Model:
        inp = layers.Input(shape=(input_dim,), name="stacked_features")
        x = layers.BatchNormalization()(inp)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(3, activation="softmax", name="predictions")(x)
        model = models.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HybridRFANN":
        """Train RF base learner then ANN meta-learner on stacked features."""
        print("[RF+ANN] Step 1/2 — Training base Random Forest ...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils.class_weight import compute_class_weight as cw

        classes = np.unique(y_train)
        weights = cw("balanced", classes=classes, y=y_train)
        cwd = dict(zip(classes.tolist(), weights.tolist()))

        self.rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            class_weight=cwd,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.rf.fit(X_train, y_train)
        os.makedirs(os.path.dirname(RF_PATH), exist_ok=True)
        joblib.dump(self.rf, RF_PATH)

        print("[RF+ANN] Step 2/2 — Building stacked features for meta-ANN ...")
        rf_proba = self.rf.predict_proba(X_train)          # [N, 3]
        X_stacked = np.hstack([X_train, rf_proba])          # [N, features + 3]

        self.ann = self._build_meta_ann(X_stacked.shape[1])
        cb_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        self.ann.fit(
            X_stacked, y_train,
            epochs=self.ann_epochs,
            batch_size=self.ann_batch_size,
            validation_split=0.1,
            class_weight=cwd,
            callbacks=cb_list,
            verbose=1,
        )
        self.ann.save(ANN_PATH)
        print("[RF+ANN] Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        rf_proba  = self.rf.predict_proba(X)
        X_stacked = np.hstack([X, rf_proba])
        proba     = self.ann.predict(X_stacked, verbose=0)
        return np.argmax(proba, axis=1).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        rf_proba  = self.rf.predict_proba(X)
        X_stacked = np.hstack([X, rf_proba])
        return self.ann.predict(X_stacked, verbose=0).astype(np.float32)

    def load(self) -> "HybridRFANN":
        self.rf  = joblib.load(RF_PATH)
        self.ann = tf.keras.models.load_model(ANN_PATH)
        print("[RF+ANN] Models loaded.")
        return self

    def _ensure_loaded(self):
        if self.rf is None or self.ann is None:
            self.load()
