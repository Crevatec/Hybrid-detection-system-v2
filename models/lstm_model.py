# models/lstm_model.py
"""
LSTM Detector — Phase 2 Individual Model.

Long Short-Term Memory network for sequence-based behavioral analysis.
Each sample is treated as a sequence by reshaping feature vectors into
time-steps (e.g., 5 time-steps × (features/5) dimensions).

Architecture:
  Input(seq_len, feat_per_step)
    → LSTM(128, return_sequences=True)
    → Dropout(0.3)
    → LSTM(64)
    → Dropout(0.2)
    → Dense(32, relu)
    → Dense(3, softmax)
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "saved_models", "lstm_model.keras"
)
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "saved_models", "lstm_config.npy"
)


class LSTMDetector:
    """
    LSTM network for temporal behavioral sequence classification.

    Parameters
    ----------
    seq_len      : number of time-steps per sequence
    num_classes  : 3
    epochs       : training epochs
    batch_size   : mini-batch size
    """

    def __init__(
        self,
        seq_len: int = 5,
        num_classes: int = 3,
        epochs: int = 25,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.seq_len       = seq_len
        self.num_classes   = num_classes
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.model         = None
        self.feat_per_step = None

        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """Reshape flat feature vector into (N, seq_len, feat_per_step)."""
        n_features = X.shape[1]
        # Pad features so they divide evenly by seq_len
        remainder = n_features % self.seq_len
        if remainder != 0:
            pad_width = self.seq_len - remainder
            X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
        self.feat_per_step = X.shape[1] // self.seq_len
        return X.reshape(X.shape[0], self.seq_len, self.feat_per_step)

    def _build(self, seq_len: int, feat_per_step: int) -> tf.keras.Model:
        inp = layers.Input(shape=(seq_len, feat_per_step), name="sequence_input")
        x = layers.LSTM(128, return_sequences=True)(inp)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)

        model = models.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LSTMDetector":
        """Train the LSTM on reshaped sequence data."""
        X_seq = self._reshape(X_train)
        print(f"[LSTM] Input shape after reshape: {X_seq.shape}  "
              f"(seq_len={self.seq_len}, feat_per_step={self.feat_per_step})")

        self.model = self._build(self.seq_len, self.feat_per_step)
        self.model.summary()

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

        cb_list = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ]

        self.model.fit(
            X_seq, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            class_weight=class_weight_dict,
            callbacks=cb_list,
            verbose=1,
        )
        print("[LSTM] Training complete.")
        self._save()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        X_seq = self._reshape(X)
        proba = self.model.predict(X_seq, verbose=0)
        return np.argmax(proba, axis=1).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        X_seq = self._reshape(X)
        return self.model.predict(X_seq, verbose=0).astype(np.float32)

    def _save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self.model.save(MODEL_PATH)
        # Save config so predict can reconstruct reshape
        np.save(CONFIG_PATH, np.array([self.seq_len, self.feat_per_step]))
        print(f"[LSTM] Model saved → {MODEL_PATH}")

    def load(self) -> "LSTMDetector":
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"LSTM model not found at {MODEL_PATH}. Run training first.")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        cfg = np.load(CONFIG_PATH)
        self.seq_len, self.feat_per_step = int(cfg[0]), int(cfg[1])
        print(f"[LSTM] Model loaded from {MODEL_PATH}")
        return self

    def _ensure_loaded(self):
        if self.model is None:
            self.load()
