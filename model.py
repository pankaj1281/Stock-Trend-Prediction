"""
model.py
--------
Builds, trains, evaluates and persists the LSTM-based stock price
prediction model.
"""

import os
import logging
import random

import numpy as np

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# TensorFlow / Keras imports (imported lazily inside functions so that
# the module can be imported even without GPU drivers)
logger = logging.getLogger(__name__)


def _set_tf_seed() -> None:
    import tensorflow as tf  # noqa: PLC0415

    tf.random.set_seed(RANDOM_SEED)


# ── Model construction ─────────────────────────────────────────────────────────


def build_lstm_model(
    input_shape: tuple[int, int],
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    dense_units: int = 25,
) -> "tf.keras.Model":
    """Build a two-layer LSTM model with dropout.

    Architecture
    ------------
    LSTM(lstm_units, return_sequences=True) → Dropout
    LSTM(lstm_units)                        → Dropout
    Dense(dense_units, relu)
    Dense(1)

    Parameters
    ----------
    input_shape:
        ``(window_size, n_features)`` – shape of one input sample (excluding
        the batch dimension).
    lstm_units:
        Number of units in each LSTM layer.
    dropout_rate:
        Dropout fraction applied after each LSTM layer.
    dense_units:
        Units in the intermediate Dense layer.

    Returns
    -------
    tf.keras.Model
        Compiled model (Adam optimizer, MSE loss, MAE metric).
    """
    _set_tf_seed()
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras.models import Sequential  # noqa: PLC0415
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # noqa: PLC0415

    model = Sequential(
        [
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ],
        name="StockLSTM",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    logger.info("Model built:\n%s", model.summary())
    return model


# ── Training ───────────────────────────────────────────────────────────────────


def train_model(
    model: "tf.keras.Model",
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.1,
    patience: int = 10,
) -> "tf.keras.callbacks.History":
    """Train *model* with early-stopping.

    Parameters
    ----------
    model:
        Compiled Keras model from :func:`build_lstm_model`.
    X_train, y_train:
        Training sequences from :func:`preprocessing.build_lstm_dataset`.
    epochs:
        Maximum training epochs.
    batch_size:
        Mini-batch size.
    validation_split:
        Fraction of training data used for validation.
    patience:
        Early-stopping patience (epochs without val_loss improvement).

    Returns
    -------
    tf.keras.callbacks.History
        Keras training history object.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: PLC0415

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(
        "Training for up to %d epochs (batch=%d, val_split=%.0f%%) …",
        epochs,
        batch_size,
        validation_split * 100,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── Evaluation ─────────────────────────────────────────────────────────────────


def evaluate_model(
    model: "tf.keras.Model",
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    feature_cols: list[str],
) -> dict:
    """Evaluate the model and return MAE / RMSE in the original price scale.

    Parameters
    ----------
    model:
        Trained Keras model.
    X_test, y_test:
        Test sequences (normalised).
    scaler:
        Fitted :class:`sklearn.preprocessing.MinMaxScaler` used during
        preprocessing (needed for inverse transform).
    feature_cols:
        Feature column names in the order they were fed to the scaler.
        ``"Close"`` must be among them (or be the first column).

    Returns
    -------
    dict
        ``{"mae": float, "rmse": float, "predictions": np.ndarray,
           "actuals": np.ndarray}``
    """
    predictions_scaled = model.predict(X_test, verbose=0)

    # Build a dummy full-width array for inverse_transform (scaler expects
    # the same number of features it was fitted on)
    n_features = len(feature_cols)
    close_idx = feature_cols.index("Close") if "Close" in feature_cols else 0

    dummy = np.zeros((len(predictions_scaled), n_features))
    dummy[:, close_idx] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy)[:, close_idx]

    dummy_actual = np.zeros((len(y_test), n_features))
    dummy_actual[:, close_idx] = y_test.flatten()
    actuals = scaler.inverse_transform(dummy_actual)[:, close_idx]

    mae = float(np.mean(np.abs(predictions - actuals)))
    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))

    logger.info("Evaluation → MAE: %.4f  RMSE: %.4f", mae, rmse)
    return {"mae": mae, "rmse": rmse, "predictions": predictions, "actuals": actuals}


# ── Persistence ────────────────────────────────────────────────────────────────


def save_model(model: "tf.keras.Model", path: str = "model.keras") -> None:
    """Save *model* to *path* (.keras or .h5 format).

    Parameters
    ----------
    model:
        Trained Keras model.
    path:
        Destination file path.  Extension determines the format:
        ``.keras`` (default) or ``.h5``.
    """
    model.save(path)
    logger.info("Model saved to '%s'.", path)


def load_model(path: str) -> "tf.keras.Model":
    """Load a previously saved Keras model.

    Parameters
    ----------
    path:
        Path to ``.keras`` or ``.h5`` file.

    Returns
    -------
    tf.keras.Model

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: '{path}'")

    import tensorflow as tf  # noqa: PLC0415

    model = tf.keras.models.load_model(path)
    logger.info("Model loaded from '%s'.", path)
    return model


# ── Prediction helper ──────────────────────────────────────────────────────────


def predict_next_close(
    model: "tf.keras.Model",
    last_window: np.ndarray,
    scaler,
    feature_cols: list[str],
) -> float:
    """Predict the next closing price given the most recent *window_size* rows.

    Parameters
    ----------
    model:
        Trained Keras model.
    last_window:
        Normalised array of shape ``(1, window_size, n_features)`` or
        ``(window_size, n_features)`` – the most recent look-back window.
    scaler:
        Fitted MinMaxScaler.
    feature_cols:
        Feature column order used during training.

    Returns
    -------
    float
        Predicted closing price in the original (dollar) scale.
    """
    if last_window.ndim == 2:
        last_window = last_window[np.newaxis, ...]

    pred_scaled = model.predict(last_window, verbose=0)

    n_features = len(feature_cols)
    close_idx = feature_cols.index("Close") if "Close" in feature_cols else 0

    dummy = np.zeros((1, n_features))
    dummy[0, close_idx] = pred_scaled[0, 0]
    price = scaler.inverse_transform(dummy)[0, close_idx]
    return float(price)
