# models/ann_model.py
"""
Artificial Neural Network (ANN) Detector — Phase 2 Individual Model.

Dense feed-forward network for multi-class classification.
Architecture:
  Input → BatchNorm → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.2)
       → Dense(64) → Dense(3, softmax)
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "saved_models", "ann_model.keras"
)


class ANNDetector:
    """
    Dense Neural Network for behavioral attack classification.

    Parameters
    ----------
    input_dim    : number of input features (set automatically on fit)
    num_classes  : 3 (Normal, CredentialStuffing, BusinessLogic)
    epochs       : training epochs
    batch_size   : mini-batch size
    learning_rate: Adam optimizer learning rate
    """

    def __init__(
        self,
        input_dim: int = None,
        num_classes: int = 3,
        epochs: int = 30,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.input_dim     = input_dim
        self.num_classes   = num_classes
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.model         = None
        self.history       = None

        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def _build(self, input_dim: int) -> tf.keras.Model:
        inp = layers.Input(shape=(input_dim,), name="features")
        x = layers.BatchNormalization()(inp)
        x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)

        model = models.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ANNDetector":
        """Train the ANN."""
        self.input_dim = X_train.shape[1]
        print(f"[ANN] Building model — input_dim={self.input_dim}, "
              f"epochs={self.epochs}, batch={self.batch_size}")
        self.model = self._build(self.input_dim)
        self.model.summary()

        # Class weights for imbalance
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

        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            class_weight=class_weight_dict,
            callbacks=cb_list,
            verbose=1,
        )
        print("[ANN] Training complete.")
        self._save()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        self._ensure_loaded()
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probability estimates [N, 3]."""
        self._ensure_loaded()
        return self.model.predict(X, verbose=0).astype(np.float32)

    def _save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self.model.save(MODEL_PATH)
        print(f"[ANN] Model saved → {MODEL_PATH}")

    def load(self) -> "ANNDetector":
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ANN model not found at {MODEL_PATH}. Run training first.")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[ANN] Model loaded from {MODEL_PATH}")
        return self

    def _ensure_loaded(self):
        if self.model is None:
            self.load()
