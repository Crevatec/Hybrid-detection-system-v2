# models/hybrid_rf_lstm.py
"""
Hybrid RF + LSTM Detector — Phase 3 Ensemble Model.

Strategy: Parallel Ensemble with Weighted Averaging
  1. Random Forest processes structured/tabular features → probability vector [N, 3]
  2. LSTM processes the same features reshaped as sequences → probability vector [N, 3]
  3. Final prediction = weighted average of both probability vectors
     (weights: RF=0.45, LSTM=0.55)
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

RF_PATH   = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "rf_lstm_rf.pkl")
LSTM_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "rf_lstm_lstm.keras")
CFG_PATH  = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "rf_lstm_config.npy")


class HybridRFLSTM:
    def __init__(self, rf_n_estimators=150, seq_len=5, lstm_epochs=25,
                 lstm_batch_size=512, rf_weight=0.45, random_state=42):
        self.rf_n_estimators = rf_n_estimators
        self.seq_len         = seq_len
        self.lstm_epochs     = lstm_epochs
        self.lstm_batch_size = lstm_batch_size
        self.rf_weight       = rf_weight
        self.lstm_weight     = 1.0 - rf_weight
        self.random_state    = random_state
        self.rf              = None
        self.lstm_model      = None
        self.feat_per_step   = None

    def _reshape(self, X):
        n_features = X.shape[1]
        remainder  = n_features % self.seq_len
        if remainder != 0:
            X = np.pad(X, ((0,0),(0, self.seq_len - remainder)), mode="constant")
        self.feat_per_step = X.shape[1] // self.seq_len
        return X.reshape(X.shape[0], self.seq_len, self.feat_per_step)

    def _build_lstm(self, seq_len, feat_per_step):
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

    def fit(self, X_train, y_train):
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cwd     = dict(zip(classes.tolist(), weights.tolist()))

        print("[RF+LSTM] Step 1/2 — Training Random Forest ...")
        self.rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators, class_weight=cwd,
            n_jobs=-1, random_state=self.random_state)
        self.rf.fit(X_train, y_train)
        os.makedirs(os.path.dirname(RF_PATH), exist_ok=True)
        joblib.dump(self.rf, RF_PATH)

        print("[RF+LSTM] Step 2/2 — Training LSTM ...")
        X_seq = self._reshape(X_train)
        self.lstm_model = self._build_lstm(self.seq_len, self.feat_per_step)
        cb_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        self.lstm_model.fit(X_seq, y_train, epochs=self.lstm_epochs,
                            batch_size=self.lstm_batch_size, validation_split=0.1,
                            class_weight=cwd, callbacks=cb_list, verbose=1)
        self.lstm_model.save(LSTM_PATH)
        np.save(CFG_PATH, np.array([self.seq_len, self.feat_per_step]))
        print("[RF+LSTM] Training complete.")
        return self

    def predict_proba(self, X):
        self._ensure_loaded()
        rf_proba   = np.array(self.rf.predict_proba(X), dtype=np.float32)
        X_seq      = self._reshape(X)
        lstm_proba = self.lstm_model.predict(X_seq, verbose=0).astype(np.float32)
        return self.rf_weight * rf_proba + self.lstm_weight * lstm_proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int32)

    def load(self):
        self.rf         = joblib.load(RF_PATH)
        self.lstm_model = tf.keras.models.load_model(LSTM_PATH)
        cfg = np.load(CFG_PATH)
        self.seq_len, self.feat_per_step = int(cfg[0]), int(cfg[1])
        print("[RF+LSTM] Models loaded.")
        return self

    def _ensure_loaded(self):
        if self.rf is None or self.lstm_model is None:
            self.load()
