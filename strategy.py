import logging

from config import (
    SYMBOL, RSI_OVERBOUGHT, RSI_OVERSOLD,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MODEL_CONFIDENCE_THRESHOLD,
    SMA_LONG,
)
from database import get_recent_candles, get_open_position
from indicators import calculate_all, get_latest_signals
from keras_advisor import get_advice

logger = logging.getLogger(__name__)


def check_stop_loss(entry_price: float, current_price: float) -> bool:
    return current_price <= entry_price * (1 - STOP_LOSS_PCT)


def check_take_profit(entry_price: float, current_price: float) -> bool:
    return current_price >= entry_price * (1 + TAKE_PROFIT_PCT)


def apply_rules(
    signals: dict,
    position: dict | None,
    model_advice: dict,
) -> tuple[str, str]:
    rsi = signals.get("rsi")
    current_price = signals.get("last_price")
    model_decision = model_advice.get("decision", "HOLD")
    model_confidence = model_advice.get("confidence", 0.0)

    if position:
        entry_price = position["price"]

        if current_price and check_stop_loss(entry_price, current_price):
            return "SELL", f"stop_loss triggered at {current_price:.4f} (entry {entry_price:.4f})"

        if current_price and check_take_profit(entry_price, current_price):
            return "SELL", f"take_profit triggered at {current_price:.4f} (entry {entry_price:.4f})"

        if model_decision == "SELL" and model_confidence >= MODEL_CONFIDENCE_THRESHOLD:
            return "SELL", f"model SELL (conf={model_confidence:.2f})"

        return "HOLD", "no_exit_conditions_met"

    # No open position — evaluate entry
    if rsi is not None and rsi > RSI_OVERBOUGHT:
        return "HOLD", f"rsi_overbought ({rsi:.1f})"

    sma_20 = signals.get("sma_20")
    sma_50 = signals.get("sma_50")
    last_price = signals.get("last_price")
    if sma_20 is not None and sma_50 is not None and (sma_20 <= sma_50 or (last_price is not None and last_price < sma_50)):
        return "HOLD", "downtrend (price or sma20 below sma50)"

    if model_decision == "BUY" and model_confidence >= MODEL_CONFIDENCE_THRESHOLD:
        return "BUY", f"model BUY (conf={model_confidence:.2f}, risk={model_advice.get('risk_level')})"

    return "HOLD", "no_entry_conditions_met"


def evaluate(timeframe: str = "5m") -> tuple[str, str, dict]:
    df = get_recent_candles(SYMBOL, timeframe, limit=200)

    if len(df) < SMA_LONG + 1:
        logger.warning(
            "Not enough candle data for %s %s: have %d rows, need %d",
            SYMBOL, timeframe, len(df), SMA_LONG + 1,
        )
        return "HOLD", "insufficient_data", {}

    df = calculate_all(df)
    signals = get_latest_signals(df)
    position = get_open_position()

    # Pass the full indicator-enriched DataFrame so the LSTM can use the sequence
    model_advice = get_advice(df, position)

    action, reason = apply_rules(signals, position, model_advice)
    logger.info("Strategy decision: %s | reason: %s | rsi=%.1f",
                action, reason, signals.get("rsi") or 0)
    return action, reason, signals
