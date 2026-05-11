import time
import logging

from config import STARTING_CAPITAL, TRADE_FEE_PCT, SYMBOL
from database import get_all_trades, get_open_position, get_latest_ticker, insert_trade
from strategy import evaluate
from telegram_notifier import notify_trade

logger = logging.getLogger(__name__)


def _replay_trades() -> tuple[float, float, float]:
    """Walk all trades from DB and compute (cash, iota_holdings, realized_pnl)."""
    cash = STARTING_CAPITAL
    holdings = 0.0
    realized_pnl = 0.0

    for trade in get_all_trades():
        action = trade["action"]
        price = trade["price"]
        qty = trade["quantity"]
        fee = trade["fee"]

        if action == "BUY":
            cash -= price * qty + fee
            holdings += qty
        elif action == "SELL":
            cash += price * qty - fee
            holdings -= qty
            realized_pnl += trade["pnl"]

    return cash, holdings, realized_pnl


def get_portfolio_status() -> dict:
    cash, holdings, realized_pnl = _replay_trades()

    ticker = get_latest_ticker(SYMBOL)
    current_price = ticker["last_price"] if ticker else 0.0

    position = get_open_position()
    entry_price = position["price"] if position else None

    unrealized_pnl = 0.0
    if position and current_price:
        qty = position["quantity"]
        buy_fee = position["fee"]
        sell_proceeds = qty * current_price
        sell_fee = sell_proceeds * TRADE_FEE_PCT
        unrealized_pnl = (sell_proceeds - sell_fee) - (qty * position["price"] + buy_fee)

    portfolio_value = cash + holdings * current_price
    all_trades = get_all_trades()

    return {
        "cash": round(cash, 4),
        "iota_holdings": round(holdings, 6),
        "entry_price": entry_price,
        "current_price": current_price,
        "portfolio_value": round(portfolio_value, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "realized_pnl": round(realized_pnl, 4),
        "total_pnl": round(realized_pnl + unrealized_pnl, 4),
        "trade_count": len(all_trades),
    }


def execute_buy(price: float, reason: str) -> dict | None:
    cash, holdings, _ = _replay_trades()

    if cash < 1.0:
        logger.warning("Insufficient cash to buy: %.4f USD", cash)
        return None

    # Fee is included within available cash so total spend == cash
    fee = cash * TRADE_FEE_PCT
    qty = (cash - fee) / price
    portfolio_value = holdings * price  # after buy, cash = 0

    trade = {
        "timestamp": int(time.time() * 1000),
        "action": "BUY",
        "price": price,
        "quantity": qty,
        "fee": fee,
        "pnl": 0.0,
        "portfolio_value": portfolio_value,
        "reason": reason,
    }
    insert_trade(trade)
    logger.info("BUY executed: qty=%.6f price=%.4f fee=%.4f reason=%s", qty, price, fee, reason)
    notify_trade("BUY", price, qty, 0.0, reason, portfolio_value)
    return trade


def execute_sell(price: float, reason: str) -> dict | None:
    position = get_open_position()
    if not position:
        logger.warning("No open position to sell")
        return None

    qty = position["quantity"]
    entry_price = position["price"]
    buy_fee = position["fee"]

    sell_proceeds = qty * price
    sell_fee = sell_proceeds * TRADE_FEE_PCT
    pnl = (sell_proceeds - sell_fee) - (qty * entry_price + buy_fee)

    cash, holdings, _ = _replay_trades()
    portfolio_value = (cash + sell_proceeds - sell_fee)

    trade = {
        "timestamp": int(time.time() * 1000),
        "action": "SELL",
        "price": price,
        "quantity": qty,
        "fee": sell_fee,
        "pnl": pnl,
        "portfolio_value": portfolio_value,
        "reason": reason,
    }
    insert_trade(trade)
    logger.info(
        "SELL executed: qty=%.6f price=%.4f fee=%.4f pnl=%.4f reason=%s",
        qty, price, sell_fee, pnl, reason,
    )
    notify_trade("SELL", price, qty, pnl, reason, portfolio_value)
    return trade


def run_iteration() -> None:
    action, reason, signals = evaluate(timeframe="5m")
    ticker = get_latest_ticker(SYMBOL)

    if not ticker:
        logger.error("No ticker data available — run 'collect' first")
        return

    current_price = ticker["last_price"]

    if action == "BUY":
        execute_buy(current_price, reason)
    elif action == "SELL":
        execute_sell(current_price, reason)
    else:
        logger.info("HOLD — no trade executed (%s)", reason)

    status = get_portfolio_status()
    print("\n=== Portfolio Status ===")
    print(f"  Cash            : ${status['cash']:.4f}")
    print(f"  IOTA Holdings   : {status['iota_holdings']:.6f}")
    print(f"  Current Price   : ${status['current_price']:.4f}")
    print(f"  Portfolio Value : ${status['portfolio_value']:.4f}")
    print(f"  Unrealized PnL  : ${status['unrealized_pnl']:.4f}")
    print(f"  Realized PnL    : ${status['realized_pnl']:.4f}")
    print(f"  Total PnL       : ${status['total_pnl']:.4f}")
    print(f"  Trade Count     : {status['trade_count']}")
    print(f"  Decision        : {action} ({reason})")
    print()
