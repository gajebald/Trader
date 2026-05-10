import time
import logging

import requests
import schedule

from config import (
    BITFINEX_BASE, SYMBOL, TIMEFRAMES, CANDLE_LIMIT,
    API_RETRY_ATTEMPTS, API_RETRY_DELAY, COLLECT_INTERVAL_SECONDS,
)
from database import insert_ticker, insert_candles

logger = logging.getLogger(__name__)


def _get_with_retry(url: str, params: dict = None) -> list:
    delay = API_RETRY_DELAY
    last_error = None
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_error = e
            logger.warning("Request failed (attempt %d/%d): %s", attempt, API_RETRY_ATTEMPTS, e)
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"All {API_RETRY_ATTEMPTS} attempts failed for {url}: {last_error}")


def fetch_ticker(symbol: str = SYMBOL) -> dict:
    url = f"{BITFINEX_BASE}/ticker/{symbol}"
    data = _get_with_retry(url)
    # [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, DAILY_CHANGE_PERC, LAST_PRICE, VOLUME, HIGH, LOW]
    return {
        "timestamp": int(time.time() * 1000),
        "symbol": symbol,
        "bid": data[0],
        "ask": data[2],
        "last_price": data[6],
        "volume": data[7],
        "high": data[8],
        "low": data[9],
        "spread": data[2] - data[0],
        "source": "bitfinex",
    }


def fetch_candles(symbol: str = SYMBOL, timeframe: str = "1m", limit: int = CANDLE_LIMIT) -> list:
    url = f"{BITFINEX_BASE}/candles/trade:{timeframe}:{symbol}/hist"
    # sort=1 returns ascending order (oldest first)
    raw = _get_with_retry(url, params={"limit": limit, "sort": 1})
    candles = []
    for c in raw:
        # Bitfinex candle format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
        # Note: CLOSE is index 2, HIGH is index 3 (not standard OHLCV order)
        candles.append({
            "timestamp": c[0],
            "symbol": symbol,
            "timeframe": timeframe,
            "open": c[1],
            "close": c[2],
            "high": c[3],
            "low": c[4],
            "volume": c[5],
        })
    return candles


def collect_once() -> None:
    logger.info("Starting collection cycle")
    try:
        ticker = fetch_ticker()
        insert_ticker(ticker)
        logger.info("Ticker saved: last_price=%.4f spread=%.4f", ticker["last_price"], ticker["spread"])
    except Exception as e:
        logger.error("Ticker fetch failed: %s", e)

    for tf in TIMEFRAMES:
        try:
            candles = fetch_candles(timeframe=tf)
            insert_candles(candles)
            logger.info("Candles saved: timeframe=%s count=%d", tf, len(candles))
        except Exception as e:
            logger.error("Candle fetch failed for %s: %s", tf, e)


def run_collection_loop() -> None:
    logger.info("Starting collection loop (interval=%ds)", COLLECT_INTERVAL_SECONDS)
    collect_once()  # run immediately on start
    schedule.every(COLLECT_INTERVAL_SECONDS).seconds.do(collect_once)
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error("Scheduler error: %s", e)
        time.sleep(1)
