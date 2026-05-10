import logging

import pandas as pd

from config import (
    SYMBOL, STARTING_CAPITAL, TRADE_FEE_PCT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    RSI_OVERBOUGHT, LLM_CONFIDENCE_THRESHOLD, SMA_LONG,
)
from database import get_recent_candles
from indicators import calculate_all

logger = logging.getLogger(__name__)


def _compute_max_drawdown(equity_curve: list) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _simulate_trades(df: pd.DataFrame) -> tuple[list, list]:
    """
    Indicator-only strategy replay — no LLM for deterministic results.
    Returns (trades_list, equity_curve).
    """
    cash = STARTING_CAPITAL
    holdings = 0.0
    entry_price = None
    buy_fee = 0.0
    trades = []
    equity_curve = []

    for _, row in df.iterrows():
        price = row["close"]
        rsi = row.get("rsi")

        if pd.isna(price) or price <= 0:
            continue

        current_equity = cash + holdings * price
        equity_curve.append(current_equity)

        # Exit logic (position open)
        if holdings > 0 and entry_price is not None:
            stop_triggered = price <= entry_price * (1 - STOP_LOSS_PCT)
            profit_triggered = price >= entry_price * (1 + TAKE_PROFIT_PCT)
            rsi_exit = rsi is not None and not pd.isna(rsi) and rsi > RSI_OVERBOUGHT

            if stop_triggered or profit_triggered or rsi_exit:
                sell_proceeds = holdings * price
                sell_fee = sell_proceeds * TRADE_FEE_PCT
                pnl = (sell_proceeds - sell_fee) - (holdings * entry_price + buy_fee)
                cash += sell_proceeds - sell_fee

                reason = "stop_loss" if stop_triggered else ("take_profit" if profit_triggered else "rsi_overbought")
                trades.append({
                    "action": "SELL",
                    "price": price,
                    "quantity": holdings,
                    "pnl": pnl,
                    "reason": reason,
                    "pnl_pct": pnl / (holdings * entry_price + buy_fee) * 100,
                })
                holdings = 0.0
                entry_price = None
                buy_fee = 0.0
                continue

        # Entry logic (no position)
        if holdings == 0:
            rsi_ok = rsi is not None and not pd.isna(rsi) and rsi <= RSI_OVERBOUGHT

            macd = row.get("macd")
            macd_signal = row.get("macd_signal")
            sma_20 = row.get("sma_20")
            sma_50 = row.get("sma_50")

            macd_bullish = (
                macd is not None and macd_signal is not None
                and not pd.isna(macd) and not pd.isna(macd_signal)
                and macd > macd_signal
            )
            trend_up = (
                sma_20 is not None and sma_50 is not None
                and not pd.isna(sma_20) and not pd.isna(sma_50)
                and sma_20 > sma_50
            )

            if rsi_ok and macd_bullish and trend_up:
                fee = cash * TRADE_FEE_PCT
                qty = (cash - fee) / price
                entry_price = price
                buy_fee = fee
                holdings = qty
                cash = 0.0

                trades.append({
                    "action": "BUY",
                    "price": price,
                    "quantity": qty,
                    "pnl": 0.0,
                    "reason": "macd_bullish+trend_up",
                    "pnl_pct": 0.0,
                })

    return trades, equity_curve


def _calculate_metrics(trades: list, equity_curve: list) -> dict:
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    trade_count = len(sell_trades)

    if trade_count == 0:
        end_capital = equity_curve[-1] if equity_curve else STARTING_CAPITAL
        return {
            "start_capital": STARTING_CAPITAL,
            "end_capital": round(end_capital, 4),
            "total_return_pct": round((end_capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2),
            "trade_count": 0,
            "win_count": 0,
            "win_rate": 0.0,
            "max_drawdown_pct": round(_compute_max_drawdown(equity_curve) * 100, 2),
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
        }

    win_count = sum(1 for t in sell_trades if t["pnl"] > 0)
    pnl_pcts = [t["pnl_pct"] for t in sell_trades]
    end_capital = equity_curve[-1] if equity_curve else STARTING_CAPITAL

    return {
        "start_capital": STARTING_CAPITAL,
        "end_capital": round(end_capital, 4),
        "total_return_pct": round((end_capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2),
        "trade_count": trade_count,
        "win_count": win_count,
        "win_rate": round(win_count / trade_count * 100, 2),
        "max_drawdown_pct": round(_compute_max_drawdown(equity_curve) * 100, 2),
        "best_trade_pct": round(max(pnl_pcts), 2),
        "worst_trade_pct": round(min(pnl_pcts), 2),
    }


def run_backtest(symbol: str = SYMBOL, timeframe: str = "1h") -> dict:
    logger.info("Starting backtest: symbol=%s timeframe=%s", symbol, timeframe)

    df = get_recent_candles(symbol, timeframe, limit=2000)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < SMA_LONG + 1:
        logger.warning("Not enough data for backtest: %d rows (need >%d)", len(df), SMA_LONG)
        return {"error": f"Insufficient data: {len(df)} rows, need >{SMA_LONG}"}

    df = calculate_all(df)

    # Drop rows where SMA_LONG is NaN (first 50 bars have incomplete indicators)
    df = df.dropna(subset=[f"sma_{SMA_LONG}"]).reset_index(drop=True)
    logger.info("Backtest data: %d usable bars after indicator warmup", len(df))

    trades, equity_curve = _simulate_trades(df)
    metrics = _calculate_metrics(trades, equity_curve)

    logger.info("Backtest complete: %s", metrics)
    return metrics
