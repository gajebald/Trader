"""
Trains an XGBoost classifier on historical IOTA candle data.

Label generation (look-ahead):
  future_return = (close[i+LOOKAHEAD] - close[i]) / close[i]
  > +LABEL_THRESHOLD_PCT  →  BUY  (class 1)
  < -LABEL_THRESHOLD_PCT  →  SELL (class 2)
  otherwise               →  HOLD (class 0)

Features: lagged FEATURE_COLUMNS at offsets defined by FEATURE_LAGS.
No sequential model needed — XGBoost handles the flat feature vector.
"""
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    SYMBOL, MODEL_PATH, MODEL_METRICS_PATH, FEATURE_LAGS, LOOKAHEAD_BARS,
    LABEL_THRESHOLD_PCT, TRAINING_EPOCHS, SMA_LONG, MIN_TRAINING_SAMPLES,
)
from database import get_recent_candles
from indicators import FEATURE_COLUMNS, calculate_all, prepare_model_features

logger = logging.getLogger(__name__)


def _generate_labels(close: pd.Series) -> np.ndarray:
    n = len(close)
    labels = np.zeros(n, dtype=int)
    for i in range(n - LOOKAHEAD_BARS):
        future_return = (close.iloc[i + LOOKAHEAD_BARS] - close.iloc[i]) / close.iloc[i]
        if future_return > LABEL_THRESHOLD_PCT:
            labels[i] = 1   # BUY
        elif future_return < -LABEL_THRESHOLD_PCT:
            labels[i] = 2   # SELL
    return labels


def _build_feature_matrix(features: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Build flat feature matrix with lagged values for XGBoost.
    For each bar i, X[i] = [features[i], features[i-lag1], features[i-lag2], ...]
    Only rows where all lags are available are included.
    """
    max_lag = max(FEATURE_LAGS)
    X, y = [], []
    for i in range(max_lag, len(features)):
        row = np.concatenate([features[i - lag] for lag in FEATURE_LAGS])
        X.append(row)
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def train(symbol: str = SYMBOL, timeframe: str = "1h") -> None:
    """Full XGBoost training pipeline: load data → features → labels → train → save."""
    import xgboost as xgb

    logger.info("Loading candle data: symbol=%s timeframe=%s", symbol, timeframe)
    df = get_recent_candles(symbol, timeframe, limit=50000)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < SMA_LONG + max(FEATURE_LAGS) + LOOKAHEAD_BARS + 1:
        raise ValueError(
            f"Not enough data: {len(df)} rows. Collect more candles and retry."
        )

    logger.info("Computing indicators on %d bars", len(df))
    df = calculate_all(df)
    df = prepare_model_features(df)

    clean = df[FEATURE_COLUMNS + ["close"]].dropna().reset_index(drop=True)
    logger.info("Clean rows after indicator warmup: %d", len(clean))

    if len(clean) < MIN_TRAINING_SAMPLES + max(FEATURE_LAGS) + LOOKAHEAD_BARS:
        raise ValueError(f"Only {len(clean)} clean rows — need more data.")

    labels = _generate_labels(clean["close"])
    features = clean[FEATURE_COLUMNS].values.astype(np.float32)

    X, y = _build_feature_matrix(features, labels)
    logger.info(
        "Training samples: %d | features per sample: %d | class dist: HOLD=%d BUY=%d SELL=%d",
        len(y), X.shape[1],
        int((y == 0).sum()), int((y == 1).sum()), int((y == 2).sum()),
    )

    if len(X) < MIN_TRAINING_SAMPLES:
        raise ValueError(f"Only {len(X)} windows after lagging. Need {MIN_TRAINING_SAMPLES}.")

    # Chronological 80/20 split — no shuffling across the split point
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Sample weights for class balancing (capped at 5× to avoid over-correction)
    n_total = len(y_train)
    counts = {cls: int((y_train == cls).sum()) for cls in [0, 1, 2]}
    weight_map = {cls: min(n_total / (3 * cnt), 5.0) if cnt > 0 else 1.0
                  for cls, cnt in counts.items()}
    sample_weight = np.array([weight_map[c] for c in y_train], dtype=np.float32)

    logger.info(
        "Class weights (capped): HOLD=%.2f BUY=%.2f SELL=%.2f",
        weight_map[0], weight_map[1], weight_map[2],
    )
    logger.info("Train: %d  Val: %d  Features: %d", len(X_train), len(X_val), X.shape[1])

    evals_result: dict = {}
    model = xgb.XGBClassifier(
        n_estimators=TRAINING_EPOCHS,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="multi:softprob",
        num_class=3,
        eval_metric=["mlogloss", "merror"],
        early_stopping_rounds=30,
        random_state=42,
        tree_method="hist",
        verbosity=1,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10,
        callbacks=[xgb.callback.EvaluationMonitor()],
    )
    # Pull evals from booster directly (works with all xgb versions)
    evals_result = model.evals_result()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info("XGBoost model saved to %s", MODEL_PATH)

    # Compute final metrics
    val_preds = model.predict(X_val)
    val_accuracy = float((val_preds == y_val).mean())
    val_loss_hist = evals_result.get("validation_1", {}).get("mlogloss", [1.0])
    val_acc_hist = [1.0 - e for e in evals_result.get("validation_1", {}).get("merror", [0.0])]
    train_loss_hist = evals_result.get("validation_0", {}).get("mlogloss", [1.0])
    train_acc_hist = [1.0 - e for e in evals_result.get("validation_0", {}).get("merror", [0.0])]

    n_rounds = model.best_iteration + 1 if hasattr(model, "best_iteration") else TRAINING_EPOCHS
    val_loss_final = float(val_loss_hist[model.best_iteration]) if hasattr(model, "best_iteration") else val_loss_hist[-1]

    # Feature importances (top features by gain)
    importance = model.get_booster().get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timeframe": timeframe,
        "epochs_ran": n_rounds,
        "val_loss": round(val_loss_final, 4),
        "val_accuracy": round(val_accuracy, 4),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "class_weights": {
            "HOLD": round(weight_map[0], 3),
            "BUY":  round(weight_map[1], 3),
            "SELL": round(weight_map[2], 3),
        },
        "feature_importances": {k: round(v, 2) for k, v in top_features},
        "history": {
            "loss":         [round(float(x), 4) for x in train_loss_hist],
            "val_loss":     [round(float(x), 4) for x in val_loss_hist],
            "accuracy":     [round(float(x), 4) for x in train_acc_hist],
            "val_accuracy": [round(float(x), 4) for x in val_acc_hist],
        },
    }
    with open(MODEL_METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", MODEL_METRICS_PATH)

    print(f"\n=== XGBoost Training Complete ===")
    print(f"  Boosting rounds : {n_rounds}")
    print(f"  Val loss        : {val_loss_final:.4f}")
    print(f"  Val accuracy    : {val_accuracy:.4f}")
    print(f"  Model saved to  : {MODEL_PATH}")
    print(f"  Class weights   : HOLD={weight_map[0]:.2f} BUY={weight_map[1]:.2f} SELL={weight_map[2]:.2f}")
    print()
