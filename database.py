import sqlite3
import logging
from contextlib import contextmanager

import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)


@contextmanager
def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def setup_database() -> None:
    with _get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_data (
                timestamp  INTEGER PRIMARY KEY,
                symbol     TEXT NOT NULL,
                bid        REAL,
                ask        REAL,
                last_price REAL,
                volume     REAL,
                high       REAL,
                low        REAL,
                spread     REAL,
                source     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candle_data (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol    TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open      REAL,
                close     REAL,
                high      REAL,
                low       REAL,
                volume    REAL,
                UNIQUE(timestamp, symbol, timeframe)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       INTEGER NOT NULL,
                action          TEXT NOT NULL,
                price           REAL NOT NULL,
                quantity        REAL NOT NULL,
                fee             REAL NOT NULL,
                pnl             REAL NOT NULL,
                portfolio_value REAL NOT NULL,
                reason          TEXT
            )
        """)
    logger.info("Database setup complete: %s", DB_PATH)


def insert_ticker(data: dict) -> None:
    with _get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO ticker_data
                (timestamp, symbol, bid, ask, last_price, volume, high, low, spread, source)
            VALUES
                (:timestamp, :symbol, :bid, :ask, :last_price, :volume, :high, :low, :spread, :source)
        """, data)


def insert_candles(candles: list) -> None:
    if not candles:
        return
    with _get_connection() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO candle_data
                (timestamp, symbol, timeframe, open, close, high, low, volume)
            VALUES
                (:timestamp, :symbol, :timeframe, :open, :close, :high, :low, :volume)
        """, candles)


def insert_trade(trade: dict) -> None:
    with _get_connection() as conn:
        conn.execute("""
            INSERT INTO trades
                (timestamp, action, price, quantity, fee, pnl, portfolio_value, reason)
            VALUES
                (:timestamp, :action, :price, :quantity, :fee, :pnl, :portfolio_value, :reason)
        """, trade)


def get_recent_candles(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT timestamp, open, close, high, low, volume
            FROM candle_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (symbol, timeframe, limit)).fetchall()

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "close", "high", "low", "volume"])

    df = pd.DataFrame([dict(r) for r in rows])
    return df.sort_values("timestamp").reset_index(drop=True)


def get_latest_ticker(symbol: str) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM ticker_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol,)).fetchone()
    return dict(row) if row else None


def get_all_trades() -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_data_stats(symbol: str) -> dict:
    """Returns collection stats per timeframe plus ticker count."""
    with _get_connection() as conn:
        ticker_count = conn.execute(
            "SELECT COUNT(*) FROM ticker_data WHERE symbol = ?", (symbol,)
        ).fetchone()[0]

        latest_ticker = conn.execute(
            "SELECT timestamp FROM ticker_data WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
            (symbol,),
        ).fetchone()

        timeframes = {}
        for tf in ["1m", "5m", "1h"]:
            row = conn.execute("""
                SELECT COUNT(*) as cnt,
                       MIN(timestamp) as first_ts,
                       MAX(timestamp) as last_ts
                FROM candle_data
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, tf)).fetchone()
            timeframes[tf] = {
                "count": row[0],
                "first_ts": row[1],
                "last_ts": row[2],
            }

    return {
        "ticker_count": ticker_count,
        "latest_ticker_ts": latest_ticker[0] if latest_ticker else None,
        "timeframes": timeframes,
    }


def get_open_position() -> dict | None:
    with _get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM trades
            WHERE action = 'BUY'
              AND id > (
                  SELECT COALESCE(MAX(id), 0)
                  FROM trades
                  WHERE action = 'SELL'
              )
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()
    return dict(row) if row else None


def reset_trades() -> int:
    """Delete all trades and return the number of deleted rows."""
    with _get_connection() as conn:
        result = conn.execute("DELETE FROM trades")
        return result.rowcount
