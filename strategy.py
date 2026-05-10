import logging

from config import (
    SYMBOL, RSI_OVERBOUGHT, RSI_OVERSOLD,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, LLM_CONFIDENCE_THRESHOLD,
    SMA_LONG,
)
from database import get_recent_candles, get_open_position
from indicators import calculate_all, get_latest_signals
from llm_advisor import get_advice

logger = logging.getLogger(__name__)


def check_stop_loss(entry_price: float, current_price: float) -> bool:
    return current_price <= entry_price * (1 - STOP_LOSS_PCT)


def check_take_profit(entry_price: float, current_price: float) -> bool:
    return current_price >= entry_price * (1 + TAKE_PROFIT_PCT)


def apply_rules(
    signals: dict,
    position: dict | None,
    llm_advice: dict,
) -> tuple[str, str]:
    rsi = signals.get("rsi")
    current_price = signals.get("last_price")
    llm_decision = llm_advice.get("decision", "HOLD")
    llm_confidence = llm_advice.get("confidence", 0.0)

    if position:
        entry_price = position["price"]

        if current_price and check_stop_loss(entry_price, current_price):
            return "SELL", f"stop_loss triggered at {current_price:.4f} (entry {entry_price:.4f})"

        if current_price and check_take_profit(entry_price, current_price):
            return "SELL", f"take_profit triggered at {current_price:.4f} (entry {entry_price:.4f})"

        if (
            rsi is not None
            and rsi > RSI_OVERBOUGHT
            and llm_decision == "SELL"
            and llm_confidence >= LLM_CONFIDENCE_THRESHOLD
        ):
            return "SELL", f"rsi_overbought ({rsi:.1f}) + LLM SELL (conf={llm_confidence:.2f})"

        return "HOLD", "no_exit_conditions_met"

    # No open position — evaluate entry
    if rsi is not None and rsi > RSI_OVERBOUGHT:
        return "HOLD", f"rsi_overbought ({rsi:.1f})"

    if llm_decision == "BUY" and llm_confidence >= LLM_CONFIDENCE_THRESHOLD:
        return "BUY", f"LLM BUY signal (conf={llm_confidence:.2f}, risk={llm_advice.get('risk_level')})"

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
    llm_advice = get_advice(signals, position)

    action, reason = apply_rules(signals, position, llm_advice)
    logger.info("Strategy decision: %s | reason: %s | rsi=%.1f",
                action, reason, signals.get("rsi") or 0)
    return action, reason, signals
