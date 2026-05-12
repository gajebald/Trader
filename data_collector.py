import time
import logging
from datetime import datetime as _dt

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


def run_historical_download(symbol: str = SYMBOL, timeframe: str = "1h", days_back: int = 365) -> int:
    """
    Bulk-download historical candles from Bitfinex going back `days_back` days.
    Paginates backwards in batches of 5000, respects the public rate limit.
    Returns total number of candles inserted (INSERT OR IGNORE — no duplicates).
    """
    _BATCH = 5000
    _DELAY = 1.2  # seconds between requests (Bitfinex public: ~90 req/min)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days_back * 24 * 3600 * 1000
    current_end = end_ms
    total = 0
    batch_num = 0

    logger.info(
        "Historical download start: %s %s | %d days back (%s → %s)",
        symbol, timeframe, days_back,
        _dt.fromtimestamp(start_ms / 1000).strftime("%Y-%m-%d"),
        _dt.fromtimestamp(end_ms / 1000).strftime("%Y-%m-%d"),
    )

    while True:
        url = f"{BITFINEX_BASE}/candles/trade:{timeframe}:{symbol}/hist"
        params = {"limit": _BATCH, "sort": -1, "end": current_end}

        try:
            raw = _get_with_retry(url, params)
        except Exception as e:
            logger.error("Batch %d failed: %s", batch_num + 1, e)
            break

        if not raw:
            break

        candles = [
            {
                "timestamp": c[0], "symbol": symbol, "timeframe": timeframe,
                "open": c[1], "close": c[2], "high": c[3], "low": c[4], "volume": c[5],
            }
            for c in raw
        ]
        insert_candles(candles)
        batch_num += 1
        total += len(raw)

        oldest_ts = min(c[0] for c in raw)
        newest_ts = max(c[0] for c in raw)
        msg = (
            f"Batch {batch_num}: {len(raw):>5} Candles "
            f"({_dt.fromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d')} – "
            f"{_dt.fromtimestamp(newest_ts/1000).strftime('%Y-%m-%d')}), "
            f"gesamt: {total}"
        )
        logger.info(msg)
        print(msg, flush=True)

        if oldest_ts <= start_ms or len(raw) < _BATCH:
            break

        current_end = oldest_ts - 1
        time.sleep(_DELAY)

    print(f"\nDownload abgeschlossen: {total} Candles für {symbol} {timeframe}", flush=True)
    logger.info("Historical download complete: %d candles for %s %s", total, symbol, timeframe)
    return total


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
