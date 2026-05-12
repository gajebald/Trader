"""
Sends Telegram notifications for trades and system events.

Configure via environment variables (or directly in config.py):
  TELEGRAM_BOT_TOKEN   — from @BotFather
  TELEGRAM_CHAT_ID     — your numeric chat ID (use @userinfobot to find it)

If either value is empty, all functions are silent no-ops.
"""
import logging
import threading

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOL

logger = logging.getLogger(__name__)

_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

_SYSTEM_LABELS = {
    "collector_start":  ("▶️", "Datensammler gestartet"),
    "collector_stop":   ("⏹️", "Datensammler gestoppt"),
    "trader_start":     ("▶️", "Trading-Loop gestartet"),
    "trader_stop":      ("⏹️", "Trading-Loop gestoppt"),
    "training_start":   ("🧠", "Model-Training gestartet"),
    "training_done":    ("✅", "Training abgeschlossen"),
    "history_start":    ("📥", "Historische Daten — Download gestartet"),
    "history_done":     ("✅", "Historische Daten — Download abgeschlossen"),
    "portfolio_reset":  ("🔄", "Portfolio zurückgesetzt"),
    "paper_run":        ("⚡", "Paper-Trading Iteration ausgeführt"),
}


def send_message(text: str) -> bool:
    """Send a raw Markdown message. Returns True on success."""
    if not _ENABLED:
        return False
    try:
        resp = requests.post(
            _API_URL,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning("Telegram API error %s: %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)
        return False


def _send_async(text: str) -> None:
    """Fire-and-forget: sends in a daemon thread so callers aren't blocked."""
    if not _ENABLED:
        return
    t = threading.Thread(target=send_message, args=(text,), daemon=True)
    t.start()


def notify_trade(action: str, price: float, quantity: float, pnl: float,
                 reason: str, portfolio_value: float) -> None:
    """Send a BUY or SELL trade notification asynchronously."""
    if not _ENABLED:
        return
    if action == "BUY":
        lines = [
            f"🟢 *BUY* — `{SYMBOL}`",
            f"💰 Preis:       `${price:.4f}`",
            f"📦 Menge:       `{quantity:.4f} IOTA`",
            f"💵 Portfolio:   `${portfolio_value:.2f}`",
            f"📝 Grund:       `{reason}`",
        ]
    else:
        sign = "+" if pnl >= 0 else ""
        trend = "📈" if pnl >= 0 else "📉"
        lines = [
            f"🔴 *SELL* — `{SYMBOL}`",
            f"💰 Preis:       `${price:.4f}`",
            f"📦 Menge:       `{quantity:.4f} IOTA`",
            f"{trend} PnL:         `{sign}${pnl:.4f}`",
            f"💵 Portfolio:   `${portfolio_value:.2f}`",
            f"📝 Grund:       `{reason}`",
        ]
    _send_async("\n".join(lines))


def notify_system(event: str, detail: str = "") -> None:
    """Send a system event notification asynchronously."""
    if not _ENABLED:
        return
    icon, label = _SYSTEM_LABELS.get(event, ("ℹ️", event))
    text = f"{icon} *{label}*"
    if detail:
        text += f"\n`{detail}`"
    _send_async(text)
