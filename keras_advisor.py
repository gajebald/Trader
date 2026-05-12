import logging
import os

import numpy as np
import pandas as pd

from config import MODEL_PATH, MODEL_CONFIDENCE_THRESHOLD, FEATURE_LAGS
from indicators import FEATURE_COLUMNS, prepare_model_features

logger = logging.getLogger(__name__)

_model = None
_IDX_TO_DECISION = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(MODEL_PATH)
        _model = m
        logger.info("XGBoost model loaded from %s", MODEL_PATH)
    except Exception as e:
        logger.error("Failed to load XGBoost model: %s", e)
    return _model


def _risk_from_confidence(confidence: float) -> str:
    if confidence >= 0.80:
        return "LOW"
    if confidence >= MODEL_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def get_advice(df: pd.DataFrame, position: dict | None = None) -> dict:
    """
    Predict BUY/SELL/HOLD using the trained XGBoost model.
    df must be indicator-enriched (output of calculate_all()).
    Returns the same dict format as the old keras_advisor for drop-in compatibility.
    """
    default_hold = {
        "decision": "HOLD",
        "confidence": 0.0,
        "reason": "model_not_ready",
        "risk_level": "HIGH",
    }

    model = _load_model()
    if model is None:
        logger.warning("No trained model found at %s — run 'python main.py train' first", MODEL_PATH)
        return default_hold

    try:
        feat_df = prepare_model_features(df)
        clean = feat_df[FEATURE_COLUMNS].dropna().values  # shape: (n, n_features)

        max_lag = max(FEATURE_LAGS)
        if len(clean) <= max_lag:
            return {**default_hold, "reason": "insufficient_data"}

        # Build flat feature vector: [features_at_t, features_at_t-1, ...]
        row = np.concatenate([clean[-(1 + lag)] for lag in FEATURE_LAGS])
        X = row.reshape(1, -1)

        probs = model.predict_proba(X)[0]   # [p_HOLD, p_BUY, p_SELL]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        decision = _IDX_TO_DECISION[idx]

        logger.info(
            "XGBoost prediction: %s (conf=%.2f) — probs HOLD=%.2f BUY=%.2f SELL=%.2f",
            decision, confidence, probs[0], probs[1], probs[2],
        )

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": f"xgboost conf={confidence:.2f}",
            "risk_level": _risk_from_confidence(confidence),
        }

    except Exception as e:
        logger.error("XGBoost prediction failed: %s", e)
        return {**default_hold, "reason": "prediction_error"}
