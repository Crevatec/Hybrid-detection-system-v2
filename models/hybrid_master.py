# models/hybrid_master.py
"""
Master Hybrid Detector: ANN + LSTM + RF — Phase 3 "Master" Ensemble.

Strategy: Tri-model Weighted Soft Voting
  1. Random Forest    → probability vector [N, 3]  (weight: 0.30)
  2. ANN              → probability vector [N, 3]  (weight: 0.35)
  3. LSTM             → probability vector [N, 3]  (weight: 0.35)
  Final = weighted average of all three probability vectors.

This is the primary detection authority on the dashboard.
All three model signals are fused to maximise detection across:
  - Structured tabular patterns (RF)
  - Non-linear feature combinations (ANN)
  - Temporal/sequential behaviour (LSTM)
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

RF_PATH   = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "master_rf.pkl")
ANN_PATH  = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "master_ann.keras")
LSTM_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "master_lstm.keras")
CFG_PATH  = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "master_config.npy")


class MasterHybrid:
    """
    ANN + LSTM + RF tri-model weighted voting ensemble.

    Parameters
    ----------
    rf_weight   : weight assigned to RF predictions
    ann_weight  : weight assigned to ANN predictions
    lstm_weight : weight assigned to LSTM predictions
                  (rf + ann + lstm weights must sum to 1.0)
    """

    def __init__(
        self,
        rf_n_estimators: int = 150,
        seq_len: int = 5,
        ann_epochs: int = 25,
        lstm_epochs: int = 20,
        batch_size: int = 1024,
        rf_weight: float = 0.30,
        ann_weight: float = 0.35,
        lstm_weight: float = 0.35,
        random_state: int = 42,
    ):
        assert abs(rf_weight + ann_weight + lstm_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.rf_n_estimators = rf_n_estimators
        self.seq_len         = seq_len
        self.ann_epochs      = ann_epochs
        self.lstm_epochs     = lstm_epochs
        self.batch_size      = batch_size
        self.rf_weight       = rf_weight
        self.ann_weight      = ann_weight
        self.lstm_weight     = lstm_weight
        self.random_state    = random_state
        self.rf              = None
        self.ann_model       = None
        self.lstm_model      = None
        self.feat_per_step   = None

    # ── Reshape helper ────────────────────────────────────────────────────────
    def _reshape(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[1]
        rem = n % self.seq_len
        if rem != 0:
            X = np.pad(X, ((0,0),(0, self.seq_len - rem)), mode="constant")
        self.feat_per_step = X.shape[1] // self.seq_len
        return X.reshape(X.shape[0], self.seq_len, self.feat_per_step)

    # ── ANN architecture ──────────────────────────────────────────────────────
    def _build_ann(self, input_dim: int) -> tf.keras.Model:
        inp = layers.Input(shape=(input_dim,))
        x = layers.BatchNormalization()(inp)
        x = layers.Dense(256, activation="relu",
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(3, activation="softmax")(x)
        m = models.Model(inputs=inp, outputs=out)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    # ── LSTM architecture ─────────────────────────────────────────────────────
    def _build_lstm(self, seq_len: int, feat_per_step: int) -> tf.keras.Model:
        inp = layers.Input(shape=(seq_len, feat_per_step))
        x = layers.LSTM(128, return_sequences=True)(inp)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(3, activation="softmax")(x)
        m = models.Model(inputs=inp, outputs=out)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MasterHybrid":
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cwd     = dict(zip(classes.tolist(), weights.tolist()))
        cb_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]

        # ── 1. Random Forest ──────────────────────────────────────────────────
        print("[Master] 1/3 — Training Random Forest ...")
        self.rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators, class_weight=cwd,
            n_jobs=-1, random_state=self.random_state)
        self.rf.fit(X_train, y_train)
        os.makedirs(os.path.dirname(RF_PATH), exist_ok=True)
        joblib.dump(self.rf, RF_PATH)

        # ── 2. ANN ────────────────────────────────────────────────────────────
        print("[Master] 2/3 — Training ANN ...")
        self.ann_model = self._build_ann(X_train.shape[1])
        self.ann_model.fit(
            X_train, y_train,
            epochs=self.ann_epochs, batch_size=self.batch_size,
            validation_split=0.1, class_weight=cwd, callbacks=cb_list, verbose=1)
        self.ann_model.save(ANN_PATH)

        # ── 3. LSTM ───────────────────────────────────────────────────────────
        print("[Master] 3/3 — Training LSTM ...")
        X_seq = self._reshape(X_train)
        self.lstm_model = self._build_lstm(self.seq_len, self.feat_per_step)
        self.lstm_model.fit(
            X_seq, y_train,
            epochs=self.lstm_epochs, batch_size=self.batch_size // 2,
            validation_split=0.1, class_weight=cwd, callbacks=cb_list, verbose=1)
        self.lstm_model.save(LSTM_PATH)
        np.save(CFG_PATH, np.array([self.seq_len, self.feat_per_step]))

        print("[Master] All three models trained and saved.")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        rf_p   = np.array(self.rf.predict_proba(X), dtype=np.float32)
        ann_p  = self.ann_model.predict(X, verbose=0).astype(np.float32)
        X_seq  = self._reshape(X)
        lstm_p = self.lstm_model.predict(X_seq, verbose=0).astype(np.float32)
        return (self.rf_weight * rf_p +
                self.ann_weight * ann_p +
                self.lstm_weight * lstm_p)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int32)

    # ── Persistence ───────────────────────────────────────────────────────────
    def load(self) -> "MasterHybrid":
        self.rf         = joblib.load(RF_PATH)
        self.ann_model  = tf.keras.models.load_model(ANN_PATH)
        self.lstm_model = tf.keras.models.load_model(LSTM_PATH)
        cfg = np.load(CFG_PATH)
        self.seq_len, self.feat_per_step = int(cfg[0]), int(cfg[1])
        print("[Master] All models loaded.")
        return self

    def _ensure_loaded(self):
        if self.rf is None or self.ann_model is None or self.lstm_model is None:
            self.load()
