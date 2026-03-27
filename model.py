"""
model.py
--------
Builds, trains, evaluates and persists deep learning models for stock price
prediction.

Three model architectures are provided:
  * :func:`build_lstm_model`   – two-layer unidirectional LSTM (baseline)
  * :func:`build_bilstm_model` – Bidirectional LSTM with Attention mechanism
  * :func:`build_random_forest_model` – scikit-learn Random Forest (benchmark)

The BiLSTM + Attention model is the recommended production architecture.
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


# ── BiLSTM + Attention model ───────────────────────────────────────────────────


def build_bilstm_model(
    input_shape: tuple[int, int],
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    num_attention_heads: int = 4,
) -> "tf.keras.Model":
    """Build a **Bidirectional LSTM** model with a **Multi-Head Attention**
    mechanism.

    Architecture
    ------------
    Input → BiLSTM(lstm_units, return_sequences=True) → Dropout
          → BiLSTM(lstm_units, return_sequences=True) → Dropout
          → MultiHeadAttention(num_attention_heads)   → LayerNorm
          → GlobalAveragePooling1D
          → Dense(dense_units, relu)                 → Dropout
          → Dense(1)

    The BiLSTM processes the sequence in both forward and backward
    directions, capturing long-range dependencies more effectively than a
    unidirectional LSTM.  The Attention layer focuses on the most informative
    time-steps before summarising to a single vector.

    Parameters
    ----------
    input_shape:
        ``(window_size, n_features)`` – shape of one input sample.
    lstm_units:
        Units in each directional LSTM (effective units = 2 × lstm_units
        because of bidirectionality).
    dropout_rate:
        Dropout fraction applied after each recurrent and dense layer.
    dense_units:
        Units in the intermediate Dense layer.
    num_attention_heads:
        Number of attention heads.  The key/value dimensionality is set to
        ``max(1, lstm_units * 2 // num_attention_heads)``.

    Returns
    -------
    tf.keras.Model
        Compiled model (Adam optimizer, MSE loss, MAE metric).
    """
    _set_tf_seed()
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras.layers import (  # noqa: PLC0415
        Bidirectional,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
        LayerNormalization,
        LSTM,
        MultiHeadAttention,
    )
    from tensorflow.keras.models import Model  # noqa: PLC0415

    # Each Bidirectional LSTM concatenates forward + backward outputs, so the
    # effective hidden size is lstm_units * 2.  Divide by num_attention_heads
    # to get the per-head key/value dimensionality.
    key_dim = max(1, (lstm_units * 2) // num_attention_heads)

    inputs = Input(shape=input_shape, name="sequence_input")

    # BiLSTM stack
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bilstm_1")(inputs)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bilstm_2")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)

    # Multi-Head Self-Attention
    attn_out = MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=key_dim,
        name="multi_head_attention",
    )(x, x)
    x = LayerNormalization(epsilon=1e-6, name="layer_norm")(x + attn_out)

    # Aggregate temporal dimension
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Feed-forward head
    x = Dense(dense_units, activation="relu", name="dense_1")(x)
    x = Dropout(dropout_rate, name="dropout_3")(x)
    outputs = Dense(1, name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="StockBiLSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    logger.info("BiLSTM+Attention model built:\n%s", model.summary())
    return model


# ── Random Forest benchmark ────────────────────────────────────────────────────


def build_random_forest_model(
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = RANDOM_SEED,
):
    """Build a Random Forest regressor as a benchmark model.

    The model expects flattened 2-D input ``(n_samples, window_size *
    n_features)`` rather than the 3-D tensor used by LSTM models.  Use
    :func:`flatten_sequences` to reshape before calling ``fit`` / ``predict``.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum depth of each tree.  ``None`` grows trees until all leaves
        are pure.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Unfitted regressor.
    """
    from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info(
        "Random Forest built (n_estimators=%d, max_depth=%s).",
        n_estimators,
        max_depth,
    )
    return rf


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten 3-D LSTM sequences ``(samples, timesteps, features)``
    to 2-D ``(samples, timesteps * features)`` for sklearn models.

    Parameters
    ----------
    X:
        3-D array from :func:`preprocessing.create_sequences`.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_samples, window_size * n_features)``.
    """
    return X.reshape(X.shape[0], -1)


# ── Training ───────────────────────────────────────────────────────────────────


def train_model(
    model: "tf.keras.Model",
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.1,
    patience: int = 10,
    checkpoint_path: str | None = None,
) -> "tf.keras.callbacks.History":
    """Train *model* with early-stopping and optional model checkpointing.

    Parameters
    ----------
    model:
        Compiled Keras model from :func:`build_lstm_model` or
        :func:`build_bilstm_model`.
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
    checkpoint_path:
        Optional path to save the best model during training (e.g.
        ``"saved_models/best_model.keras"``).  When *None*, no checkpoint
        is written.

    Returns
    -------
    tf.keras.callbacks.History
        Keras training history object.
    """
    from tensorflow.keras.callbacks import (  # noqa: PLC0415
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

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

    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )
        logger.info("Model checkpointing enabled → '%s'.", checkpoint_path)

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

    # R² score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    logger.info("Evaluation → MAE: %.4f  RMSE: %.4f  R²: %.4f", mae, rmse, r2)
    return {"mae": mae, "rmse": rmse, "r2": r2, "predictions": predictions, "actuals": actuals}


def evaluate_rf_model(
    rf_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    feature_cols: list[str],
) -> dict:
    """Evaluate a Random Forest model and return MAE / RMSE / R².

    Parameters
    ----------
    rf_model:
        Fitted ``sklearn.ensemble.RandomForestRegressor``.
    X_test:
        3-D test array (will be flattened automatically).
    y_test:
        Normalised target values.
    scaler, feature_cols:
        Same as :func:`evaluate_model`.

    Returns
    -------
    dict
        ``{"mae": float, "rmse": float, "r2": float, "predictions":
        np.ndarray, "actuals": np.ndarray}``
    """
    X_flat = flatten_sequences(X_test)
    preds_scaled = rf_model.predict(X_flat).reshape(-1, 1)

    n_features = len(feature_cols)
    close_idx = feature_cols.index("Close") if "Close" in feature_cols else 0

    dummy = np.zeros((len(preds_scaled), n_features))
    dummy[:, close_idx] = preds_scaled.flatten()
    predictions = scaler.inverse_transform(dummy)[:, close_idx]

    dummy_actual = np.zeros((len(y_test), n_features))
    dummy_actual[:, close_idx] = y_test.flatten()
    actuals = scaler.inverse_transform(dummy_actual)[:, close_idx]

    mae = float(np.mean(np.abs(predictions - actuals)))
    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    logger.info("RF Evaluation → MAE: %.4f  RMSE: %.4f  R²: %.4f", mae, rmse, r2)
    return {"mae": mae, "rmse": rmse, "r2": r2, "predictions": predictions, "actuals": actuals}


def compare_models(results: dict[str, dict]) -> "pd.DataFrame":
    """Build a summary comparison table for multiple model evaluation results.

    Parameters
    ----------
    results:
        Mapping of ``model_name → eval_dict`` where each *eval_dict* has
        keys ``mae``, ``rmse``, ``r2``.

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows.
    """
    import pandas as pd  # noqa: PLC0415

    rows = []
    for name, res in results.items():
        rows.append(
            {
                "Model": name,
                "MAE": round(res.get("mae", float("nan")), 4),
                "RMSE": round(res.get("rmse", float("nan")), 4),
                "R²": round(res.get("r2", float("nan")), 4),
            }
        )
    df = pd.DataFrame(rows).set_index("Model")
    logger.info("Model comparison:\n%s", df.to_string())
    return df


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
