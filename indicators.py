import logging

import pandas as pd

from config import (
    SMA_SHORT, SMA_LONG, EMA_PERIOD, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, VOLATILITY_WINDOW,
)

# Ordered list of feature columns used by the Keras model.
# Order matters — must be identical between training and inference.
FEATURE_COLUMNS = [
    "rsi_norm",
    "price_vs_sma20",
    "price_vs_sma50",
    "sma20_vs_sma50",
    "price_vs_ema20",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "volatility_norm",
    "pct_change_clipped",
    "high_low_ratio",
    "volume_ratio",
    "bb_position",
    "bb_width_norm",
    "rsi_slope",
]

logger = logging.getLogger(__name__)


def add_sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    df[f"sma_{period}"] = df[col].rolling(window=period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    df[f"ema_{period}"] = df[col].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
    df["volatility"] = df["close"].rolling(window=window).std()
    return df


def add_volume_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["volume_sma_20"] = df["volume"].rolling(window=window).mean()
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20, std: float = 2.0) -> pd.DataFrame:
    rolling = df["close"].rolling(window=window)
    df["bb_mid"] = rolling.mean()
    bb_std = rolling.std()
    df["bb_upper"] = df["bb_mid"] + std * bb_std
    df["bb_lower"] = df["bb_mid"] - std * bb_std
    return df


def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_sma(df, SMA_SHORT)
    df = add_sma(df, SMA_LONG)
    df = add_ema(df, EMA_PERIOD)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_volatility(df)
    df = add_volume_sma(df)
    df = add_bollinger(df)
    df["pct_change"] = df["close"].pct_change()
    return df


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute scale-invariant normalized features for the Keras model.
    Must be called on a DataFrame already enriched by calculate_all().
    Returns a new DataFrame with FEATURE_COLUMNS added (NaN where inputs are missing).
    """
    df = df.copy()
    price = df["close"].replace(0, float("nan"))
    sma20 = df[f"sma_{SMA_SHORT}"].replace(0, float("nan"))
    sma50 = df[f"sma_{SMA_LONG}"].replace(0, float("nan"))
    low = df["low"].replace(0, float("nan"))

    df["rsi_norm"] = df["rsi"] / 100.0
    df["price_vs_sma20"] = (price - sma20) / sma20
    df["price_vs_sma50"] = (price - sma50) / sma50
    df["sma20_vs_sma50"] = (sma20 - sma50) / sma50
    df["price_vs_ema20"] = (price - df[f"ema_{EMA_PERIOD}"]) / df[f"ema_{EMA_PERIOD}"].replace(0, float("nan"))
    df["macd_norm"] = df["macd"] / price
    df["macd_signal_norm"] = df["macd_signal"] / price
    df["macd_hist_norm"] = df["macd_hist"] / price
    df["volatility_norm"] = df["volatility"] / price
    df["pct_change_clipped"] = df["pct_change"].clip(-0.1, 0.1)
    df["high_low_ratio"] = (df["high"] - df["low"]) / low

    vol_sma = df["volume_sma_20"].replace(0, float("nan"))
    df["volume_ratio"] = (df["volume"] / vol_sma).clip(0, 5)

    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, float("nan"))
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range
    df["bb_width_norm"] = bb_range / df["bb_mid"].replace(0, float("nan"))

    df["rsi_slope"] = (df["rsi"].diff(3) / 30).clip(-1, 1)

    return df


def get_latest_signals(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    row = df.iloc[-1]

    def _f(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return {
        "last_price": _f(row.get("close")),
        "sma_20": _f(row.get(f"sma_{SMA_SHORT}")),
        "sma_50": _f(row.get(f"sma_{SMA_LONG}")),
        "ema_20": _f(row.get(f"ema_{EMA_PERIOD}")),
        "rsi": _f(row.get("rsi")),
        "macd": _f(row.get("macd")),
        "macd_signal": _f(row.get("macd_signal")),
        "macd_hist": _f(row.get("macd_hist")),
        "volatility": _f(row.get("volatility")),
        "pct_change": _f(row.get("pct_change")),
        "high": _f(row.get("high")),
        "low": _f(row.get("low")),
        "volume": _f(row.get("volume")),
    }
