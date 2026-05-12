"""
Trains an XGBoost classifier on historical IOTA candle data.

Label generation (look-ahead):
  future_return = (close[i+LOOKAHEAD] - close[i]) / close[i]
  > +LABEL_THRESHOLD_PCT  →  BUY  (class 1)
  < -LABEL_THRESHOLD_PCT  →  SELL (class 2)
  otherwise               →  HOLD (class 0)

Features: lagged FEATURE_COLUMNS at offsets defined by FEATURE_LAGS.
"""
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    SYMBOL, MODEL_PATH, MODEL_METRICS_PATH, BEST_PARAMS_PATH,
    FEATURE_LAGS, LOOKAHEAD_BARS, LABEL_THRESHOLD_PCT,
    TRAINING_EPOCHS, SMA_LONG, MIN_TRAINING_SAMPLES,
    WALK_FORWARD_FOLDS, OPTUNA_TRIALS,
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


def _load_best_params() -> dict:
    if not os.path.exists(BEST_PARAMS_PATH):
        return {}
    try:
        with open(BEST_PARAMS_PATH) as f:
            params = json.load(f)
        params.pop("best_val_accuracy", None)
        logger.info("Loaded tuned hyperparameters from %s", BEST_PARAMS_PATH)
        return params
    except Exception as e:
        logger.warning("Could not load best params: %s", e)
        return {}


def _build_sample_weights(y_train: np.ndarray) -> np.ndarray:
    n_total = len(y_train)
    counts = {cls: int((y_train == cls).sum()) for cls in [0, 1, 2]}
    weight_map = {cls: min(n_total / (3 * cnt), 5.0) if cnt > 0 else 1.0
                  for cls, cnt in counts.items()}
    return np.array([weight_map[c] for c in y_train], dtype=np.float32), weight_map


def _make_classifier(**extra_params) -> "xgb.XGBClassifier":
    import xgboost as xgb
    defaults = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
    }
    defaults.update(extra_params)
    return xgb.XGBClassifier(
        n_estimators=TRAINING_EPOCHS,
        objective="multi:softprob",
        num_class=3,
        eval_metric=["mlogloss", "merror"],
        early_stopping_rounds=30,
        random_state=42,
        tree_method="hist",
        verbosity=1,
        **defaults,
    )


def _load_and_prepare(symbol: str, timeframe: str) -> tuple:
    """Shared data-loading pipeline for train / validate / tune."""
    df = get_recent_candles(symbol, timeframe, limit=50000)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < SMA_LONG + max(FEATURE_LAGS) + LOOKAHEAD_BARS + 1:
        raise ValueError(f"Not enough data: {len(df)} rows.")

    df = calculate_all(df)
    df = prepare_model_features(df)
    clean = df[FEATURE_COLUMNS + ["close"]].dropna().reset_index(drop=True)

    labels = _generate_labels(clean["close"])
    features = clean[FEATURE_COLUMNS].values.astype(np.float32)
    X, y = _build_feature_matrix(features, labels)
    return X, y


def train(symbol: str = SYMBOL, timeframe: str = "1h") -> None:
    """Full XGBoost training pipeline: load data → features → labels → train → save."""
    logger.info("Loading candle data: symbol=%s timeframe=%s", symbol, timeframe)
    X, y = _load_and_prepare(symbol, timeframe)

    logger.info(
        "Training samples: %d | features per sample: %d | class dist: HOLD=%d BUY=%d SELL=%d",
        len(y), X.shape[1],
        int((y == 0).sum()), int((y == 1).sum()), int((y == 2).sum()),
    )

    if len(X) < MIN_TRAINING_SAMPLES:
        raise ValueError(f"Only {len(X)} windows after lagging. Need {MIN_TRAINING_SAMPLES}.")

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    sample_weight, weight_map = _build_sample_weights(y_train)
    logger.info(
        "Class weights (capped): HOLD=%.2f BUY=%.2f SELL=%.2f",
        weight_map[0], weight_map[1], weight_map[2],
    )
    logger.info("Train: %d  Val: %d  Features: %d", len(X_train), len(X_val), X.shape[1])

    tuned = _load_best_params()
    model = _make_classifier(**tuned)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10,
    )
    evals_result = model.evals_result()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info("XGBoost model saved to %s", MODEL_PATH)

    val_preds = model.predict(X_val)
    val_accuracy = float((val_preds == y_val).mean())
    val_loss_hist = evals_result.get("validation_1", {}).get("mlogloss", [1.0])
    val_acc_hist = [1.0 - e for e in evals_result.get("validation_1", {}).get("merror", [0.0])]
    train_loss_hist = evals_result.get("validation_0", {}).get("mlogloss", [1.0])
    train_acc_hist = [1.0 - e for e in evals_result.get("validation_0", {}).get("merror", [0.0])]

    n_rounds = model.best_iteration + 1 if hasattr(model, "best_iteration") else TRAINING_EPOCHS
    val_loss_final = float(val_loss_hist[model.best_iteration]) if hasattr(model, "best_iteration") else val_loss_hist[-1]

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
    if tuned:
        print(f"  Using tuned params from {BEST_PARAMS_PATH}")
    print()


def validate_walk_forward(symbol: str = SYMBOL, timeframe: str = "1h",
                           n_folds: int = WALK_FORWARD_FOLDS) -> list:
    """
    Honest out-of-sample evaluation via walk-forward cross-validation.
    Trains on folds 0..k, tests on fold k+1. No model is saved.
    """
    logger.info("Walk-forward validation: symbol=%s timeframe=%s folds=%d", symbol, timeframe, n_folds)
    X, y = _load_and_prepare(symbol, timeframe)

    fold_size = len(X) // n_folds
    if fold_size < MIN_TRAINING_SAMPLES:
        raise ValueError(f"Too few samples per fold ({fold_size}). Need more data or fewer folds.")

    tuned = _load_best_params()
    results = []

    for fold in range(1, n_folds):
        train_end = fold * fold_size
        val_start = train_end
        val_end = min((fold + 1) * fold_size, len(X))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        if len(X_val) == 0:
            break

        sw, _ = _build_sample_weights(y_train)

        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=TRAINING_EPOCHS,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            random_state=42,
            tree_method="hist",
            verbosity=0,
            **({**{"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8,
                   "colsample_bytree": 0.8, "min_child_weight": 5}, **tuned}),
        )
        model.fit(X_train, y_train, sample_weight=sw,
                  eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        acc = float((preds == y_val).mean())
        results.append(acc)

        dist = {cls: int((y_val == cls).sum()) for cls in [0, 1, 2]}
        print(f"  Fold {fold}/{n_folds-1}  train={len(X_train):>6}  val={len(X_val):>5}  "
              f"OOS acc={acc:.4f}  HOLD={dist[0]} BUY={dist[1]} SELL={dist[2]}")

    avg = sum(results) / len(results) if results else 0.0
    print(f"\n  Average OOS accuracy : {avg:.4f}  (random baseline = {1/3:.4f})")
    return results


def tune(symbol: str = SYMBOL, timeframe: str = "1h",
         n_trials: int = OPTUNA_TRIALS) -> None:
    """
    Automated hyperparameter search using Optuna (3-fold walk-forward objective).
    Saves best params to BEST_PARAMS_PATH for use by train().
    """
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Run: pip install optuna")
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("Loading data for Optuna tuning: symbol=%s timeframe=%s", symbol, timeframe)
    X, y = _load_and_prepare(symbol, timeframe)

    n_folds = 3
    fold_size = len(X) // n_folds

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        accs = []
        for fold in range(1, n_folds):
            X_tr, y_tr = X[:fold * fold_size], y[:fold * fold_size]
            X_v, y_v = X[fold * fold_size:(fold + 1) * fold_size], y[fold * fold_size:(fold + 1) * fold_size]
            if len(X_v) == 0:
                continue
            sw, _ = _build_sample_weights(y_tr)
            import xgboost as xgb
            m = xgb.XGBClassifier(
                n_estimators=100,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                early_stopping_rounds=20,
                random_state=42,
                tree_method="hist",
                verbosity=0,
                **params,
            )
            m.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_v, y_v)], verbose=False)
            accs.append(float((m.predict(X_v) == y_v).mean()))
        return sum(accs) / len(accs) if accs else 0.0

    print(f"\nOptuna Hyperparameter Tuning")
    print(f"  Trials     : {n_trials}")
    print(f"  CV folds   : {n_folds} (walk-forward)")
    print(f"  Samples    : {len(X)}")
    print()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {**study.best_params, "best_val_accuracy": round(study.best_value, 4)}
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n=== Tuning Complete ===")
    print(f"  Best OOS accuracy : {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"  Saved to : {BEST_PARAMS_PATH}")
    print(f"\nRun 'python main.py train' to train with best params.")
    print()
