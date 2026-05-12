"""
Trains a Keras LSTM model on historical IOTA candle data.

Label generation (look-ahead):
  future_return = (close[i+LOOKAHEAD] - close[i]) / close[i]
  > +LABEL_THRESHOLD_PCT  →  BUY  (class 1)
  < -LABEL_THRESHOLD_PCT  →  SELL (class 2)
  otherwise               →  HOLD (class 0)

The model never sees future data during inference — only the past
LOOKBACK_STEPS bars of normalized indicator features.
"""
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    SYMBOL, MODEL_PATH, MODEL_METRICS_PATH, LOOKBACK_STEPS, LOOKAHEAD_BARS,
    LABEL_THRESHOLD_PCT, TRAINING_EPOCHS, TRAINING_BATCH_SIZE,
    MIN_TRAINING_SAMPLES, SMA_LONG,
)
from database import get_recent_candles
from indicators import FEATURE_COLUMNS, calculate_all, prepare_model_features

logger = logging.getLogger(__name__)


def _generate_labels(close: pd.Series) -> np.ndarray:
    """
    Compute forward-looking labels for each bar.
    Bars within LOOKAHEAD_BARS of the end are assigned HOLD (0)
    because their future return cannot be fully observed.
    """
    n = len(close)
    labels = np.zeros(n, dtype=int)  # default HOLD
    for i in range(n - LOOKAHEAD_BARS):
        future_return = (close.iloc[i + LOOKAHEAD_BARS] - close.iloc[i]) / close.iloc[i]
        if future_return > LABEL_THRESHOLD_PCT:
            labels[i] = 1   # BUY
        elif future_return < -LABEL_THRESHOLD_PCT:
            labels[i] = 2   # SELL
    return labels


def _create_sequences(features: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Slide a window of LOOKBACK_STEPS over the feature matrix.
    For window ending at position i, label is labels[i].
    Requires: features and labels have the same length.
    """
    X, y = [], []
    for i in range(LOOKBACK_STEPS, len(features)):
        X.append(features[i - LOOKBACK_STEPS:i])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def build_model(n_features: int):
    """Two-layer GRU with BatchNorm for time-series classification into 3 classes."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    l2 = tf.keras.regularizers.l2(0.001)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK_STEPS, n_features)),
        tf.keras.layers.GRU(64, return_sequences=True, recurrent_dropout=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(32, return_sequences=False, recurrent_dropout=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Dense(3, activation="softmax"),  # HOLD, BUY, SELL
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(symbol: str = SYMBOL, timeframe: str = "1h") -> None:
    """Full training pipeline: load data → features → labels → train → save."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    logger.info("Loading candle data: symbol=%s timeframe=%s", symbol, timeframe)
    df = get_recent_candles(symbol, timeframe, limit=50000)  # load ALL available data
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < SMA_LONG + LOOKBACK_STEPS + LOOKAHEAD_BARS + 1:
        raise ValueError(
            f"Not enough data: {len(df)} rows. "
            f"Collect more candles with 'python main.py collect' and retry."
        )

    logger.info("Computing indicators on %d bars", len(df))
    df = calculate_all(df)
    df = prepare_model_features(df)

    # Drop rows where any feature or the close price is NaN
    clean = df[FEATURE_COLUMNS + ["close"]].dropna()
    logger.info("Clean rows after indicator warmup: %d", len(clean))

    if len(clean) < MIN_TRAINING_SAMPLES + LOOKBACK_STEPS + LOOKAHEAD_BARS:
        raise ValueError(
            f"Only {len(clean)} clean rows available. Need at least "
            f"{MIN_TRAINING_SAMPLES + LOOKBACK_STEPS + LOOKAHEAD_BARS}. "
            f"Run 'python main.py collect' longer and retry."
        )

    labels = _generate_labels(clean["close"])
    features = clean[FEATURE_COLUMNS].values.astype(np.float32)

    X, y = _create_sequences(features, labels)
    logger.info(
        "Training sequences: %d | class distribution: HOLD=%d BUY=%d SELL=%d",
        len(y),
        int((y == 0).sum()), int((y == 1).sum()), int((y == 2).sum()),
    )

    if len(X) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Only {len(X)} training windows after sequencing. Need {MIN_TRAINING_SAMPLES}."
        )

    # Compute class weights — capped at 5.0 to prevent over-correction
    n_total = len(y)
    class_weight = {}
    for cls in [0, 1, 2]:
        count = int((y == cls).sum())
        raw = n_total / (3 * count) if count > 0 else 1.0
        class_weight[cls] = min(raw, 5.0)

    logger.info(
        "Class weights (capped): HOLD=%.2f BUY=%.2f SELL=%.2f",
        class_weight[0], class_weight[1], class_weight[2],
    )

    # 80/20 train/validation split (time-ordered — no shuffling across the split)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info(
        "Train size: %d | Val size: %d | Features: %d",
        len(X_train), len(X_val), len(FEATURE_COLUMNS),
    )

    model = build_model(len(FEATURE_COLUMNS))
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    # Report final metrics
    val_loss = history.history["val_loss"][-1]
    val_acc = history.history["val_accuracy"][-1]
    epochs_run = len(history.history["loss"])

    # Save metrics JSON for the dashboard
    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timeframe": timeframe,
        "epochs_ran": epochs_run,
        "val_loss": round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "class_weights": {
            "HOLD": round(class_weight[0], 3),
            "BUY": round(class_weight[1], 3),
            "SELL": round(class_weight[2], 3),
        },
        "history": {
            "loss":         [round(float(x), 4) for x in history.history["loss"]],
            "val_loss":     [round(float(x), 4) for x in history.history["val_loss"]],
            "accuracy":     [round(float(x), 4) for x in history.history["accuracy"]],
            "val_accuracy": [round(float(x), 4) for x in history.history["val_accuracy"]],
        },
    }
    with open(MODEL_METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", MODEL_METRICS_PATH)

    print(f"\n=== Training Complete ===")
    print(f"  Epochs ran      : {epochs_run}")
    print(f"  Val loss        : {val_loss:.4f}")
    print(f"  Val accuracy    : {val_acc:.4f}")
    print(f"  Model saved to  : {MODEL_PATH}")
    print(f"  Class weights   : HOLD={class_weight[0]:.2f} BUY={class_weight[1]:.2f} SELL={class_weight[2]:.2f}")
    print()
