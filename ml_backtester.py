"""
ML-based backtest: slides the trained LSTM over historical candles,
applies the same strategy rules as paper_trader.py, and returns
price + portfolio equity + BUY/SELL markers for charting.
"""
import logging

import numpy as np
from datetime import datetime

from config import (
    SYMBOL, STARTING_CAPITAL, TRADE_FEE_PCT, LOOKBACK_STEPS,
    MODEL_CONFIDENCE_THRESHOLD, RSI_OVERBOUGHT, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MODEL_PATH,
)
from database import get_recent_candles
from indicators import FEATURE_COLUMNS, calculate_all, prepare_model_features

logger = logging.getLogger(__name__)

_IDX_TO_DECISION = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _load_model():
    import os
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("ML backtest: model loaded from %s", MODEL_PATH)
        return model
    except Exception as e:
        logger.error("ML backtest: failed to load model: %s", e)
        return None


def run_ml_backtest(symbol: str = SYMBOL, timeframe: str = "1h",
                    limit: int = 500, threshold: float | None = None) -> dict:
    """
    Batch-predict all windows in one model.predict() call, then
    replay strategy rules step-by-step. Returns chart-ready JSON data.
    """
    if threshold is None:
        threshold = MODEL_CONFIDENCE_THRESHOLD
    model = _load_model()
    if model is None:
        return {"error": "Kein trainiertes Modell gefunden — bitte zuerst Training starten."}

    df = get_recent_candles(symbol, timeframe, limit=limit)
    if df.empty or len(df) < LOOKBACK_STEPS + 10:
        return {"error": f"Nicht genug Daten ({len(df)} Candles) — bitte zuerst Daten sammeln."}

    df = calculate_all(df)
    df = prepare_model_features(df)
    clean = df.dropna(subset=FEATURE_COLUMNS + ["close", "rsi"]).reset_index(drop=True)
    n = len(clean)

    if n < LOOKBACK_STEPS + 10:
        return {"error": f"Nicht genug saubere Zeilen nach Indikator-Warmup ({n})."}

    features   = clean[FEATURE_COLUMNS].values.astype(np.float32)
    closes_arr = clean["close"].values
    rsis_arr   = clean["rsi"].values
    sma20_arr  = clean["sma_20"].values
    sma50_arr  = clean["sma_50"].values
    timestamps = clean["timestamp"].values
    n_windows  = n - LOOKBACK_STEPS

    # Batch all prediction windows in one forward pass
    X = np.stack([features[i : i + LOOKBACK_STEPS] for i in range(n_windows)])
    logger.info("ML backtest: predicting %d windows (timeframe=%s)", n_windows, timeframe)
    probs = model.predict(X, verbose=0)   # shape: (n_windows, 3)

    # Step-by-step trade simulation
    cash         = STARTING_CAPITAL
    holdings     = 0.0
    entry_price  = None
    entry_fee    = 0.0
    trades       = []
    equity_curve = []
    buy_prices   = []
    sell_prices  = []

    for i in range(n_windows):
        bar   = LOOKBACK_STEPS + i
        price = float(closes_arr[bar])
        rsi   = float(rsis_arr[bar])
        sma20 = float(sma20_arr[bar])
        sma50 = float(sma50_arr[bar])

        pred_idx   = int(np.argmax(probs[i]))
        confidence = float(probs[i][pred_idx])
        decision   = _IDX_TO_DECISION[pred_idx]

        action = "HOLD"
        reason = ""

        if holdings > 0 and entry_price is not None:
            if price <= entry_price * (1 - STOP_LOSS_PCT):
                action, reason = "SELL", "stop_loss"
            elif price >= entry_price * (1 + TAKE_PROFIT_PCT):
                action, reason = "SELL", "take_profit"
            elif decision == "SELL" and confidence >= threshold:
                action, reason = "SELL", f"model ({confidence:.0%})"
        else:
            if rsi <= RSI_OVERBOUGHT and sma20 > sma50 and decision == "BUY" and confidence >= threshold:
                action, reason = "BUY", f"model ({confidence:.0%})"

        if action == "BUY" and cash > 1.0:
            fee       = cash * TRADE_FEE_PCT
            qty       = (cash - fee) / price
            entry_price = price
            entry_fee   = fee
            holdings    = qty
            cash        = 0.0
            trades.append({"type": "BUY", "price": price, "reason": reason})
            buy_prices.append(price)
            sell_prices.append(None)

        elif action == "SELL" and holdings > 0:
            proceeds = holdings * price
            fee      = proceeds * TRADE_FEE_PCT
            pnl      = (proceeds - fee) - (holdings * entry_price + entry_fee)
            cash     = proceeds - fee
            trades.append({"type": "SELL", "price": price, "pnl": round(pnl, 4), "reason": reason})
            sell_prices.append(price)
            buy_prices.append(None)
            holdings    = 0.0
            entry_price = None

        else:
            buy_prices.append(None)
            sell_prices.append(None)

        equity_curve.append(round(cash + holdings * price, 4))

    labels = [
        datetime.fromtimestamp(int(ts) / 1000).strftime("%d.%m %H:%M")
        for ts in timestamps[LOOKBACK_STEPS:]
    ]
    closes_out = [round(float(closes_arr[LOOKBACK_STEPS + i]), 4) for i in range(n_windows)]

    sells    = [t for t in trades if t["type"] == "SELL"]
    wins     = [t for t in sells  if t.get("pnl", 0) > 0]
    end_cap  = equity_curve[-1] if equity_curve else STARTING_CAPITAL
    ret_pct  = (end_cap - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    logger.info(
        "ML backtest done: %d trades, return=%.2f%%, end=$%.2f",
        len(trades), ret_pct, end_cap,
    )

    return {
        "labels":      labels,
        "closes":      closes_out,
        "equity":      equity_curve,
        "buy_prices":  buy_prices,
        "sell_prices": sell_prices,
        "summary": {
            "start_capital":    round(STARTING_CAPITAL, 2),
            "end_capital":      round(end_cap, 2),
            "total_return_pct": round(ret_pct, 2),
            "trade_count":      len(trades),
            "sell_count":       len(sells),
            "win_count":        len(wins),
            "win_rate":         round(len(wins) / len(sells) * 100, 1) if sells else 0.0,
            "candles_analyzed": n_windows,
        },
    }
