import logging
import os

import numpy as np
import pandas as pd

from config import MODEL_PATH, LOOKBACK_STEPS, MODEL_CONFIDENCE_THRESHOLD
from indicators import FEATURE_COLUMNS, prepare_model_features

logger = logging.getLogger(__name__)

# Lazy-loaded singleton — model is loaded once on first prediction call
_model = None

# Maps model output index → trading decision string
_IDX_TO_DECISION = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Keras model loaded from %s", MODEL_PATH)
    except Exception as e:
        logger.error("Failed to load Keras model: %s", e)
    return _model


def _risk_from_confidence(confidence: float) -> str:
    if confidence >= 0.80:
        return "LOW"
    if confidence >= MODEL_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def get_advice(df: pd.DataFrame, position: dict | None = None) -> dict:
    """
    Predict BUY/SELL/HOLD using the trained Keras LSTM model.
    df must be indicator-enriched (output of calculate_all()).
    Returns the same dict format as the old llm_advisor for drop-in compatibility.
    Falls back to HOLD when the model is not trained yet or prediction fails.
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
        clean = feat_df[FEATURE_COLUMNS].dropna()

        if len(clean) < LOOKBACK_STEPS:
            logger.warning(
                "Not enough clean feature rows for prediction: have %d, need %d",
                len(clean), LOOKBACK_STEPS,
            )
            return {**default_hold, "reason": "insufficient_data"}

        # Take the most recent LOOKBACK_STEPS bars as the input window
        window = clean.values[-LOOKBACK_STEPS:]       # shape: (LOOKBACK_STEPS, n_features)
        X = window.reshape(1, LOOKBACK_STEPS, len(FEATURE_COLUMNS))  # (1, steps, features)

        probs = model.predict(X, verbose=0)[0]        # shape: (3,)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        decision = _IDX_TO_DECISION[idx]

        logger.info(
            "Keras prediction: %s (conf=%.2f) — probs HOLD=%.2f BUY=%.2f SELL=%.2f",
            decision, confidence, probs[0], probs[1], probs[2],
        )

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": f"keras_lstm conf={confidence:.2f}",
            "risk_level": _risk_from_confidence(confidence),
        }

    except Exception as e:
        logger.error("Keras prediction failed: %s", e)
        return {**default_hold, "reason": "prediction_error"}
