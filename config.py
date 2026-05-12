import os

# --- Bitfinex API ---
BITFINEX_BASE = "https://api-pub.bitfinex.com/v2"
SYMBOL = "tIOTUSD"
TIMEFRAMES = ["1m", "5m", "1h"]
CANDLE_LIMIT = 500
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2.0  # seconds, doubles each retry

# --- Database ---
DB_PATH = os.environ.get("TRADER_DB_PATH", "trader.db")

# --- Indicator parameters ---
SMA_SHORT = 20
SMA_LONG = 50
EMA_PERIOD = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLATILITY_WINDOW = 20

# --- Strategy thresholds ---
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOP_LOSS_PCT = 0.015     # 1.5% below entry price
TAKE_PROFIT_PCT = 0.03    # 3% above entry price
TRADE_FEE_PCT = 0.002     # 0.2% per trade
MODEL_CONFIDENCE_THRESHOLD = 0.6

# --- Paper trading ---
STARTING_CAPITAL = 1000.0  # USD

# --- Keras LSTM model ---
MODEL_PATH = "models/trading_model.ubj"
MODEL_METRICS_PATH = "models/training_metrics.json"
LOOKBACK_STEPS = 48          # kept for ml_backtester compatibility
FEATURE_LAGS = [0, 1, 2, 3, 6, 12, 24]  # bars lookback for XGBoost flat features
LOOKAHEAD_BARS = 12          # bars ahead for labels (12h on 1h data)
LABEL_THRESHOLD_PCT = 0.015  # 1.5% future move → BUY or SELL
TRAINING_EPOCHS = 150        # max boosting rounds (early stopping applies)
TRAINING_BATCH_SIZE = 32
MIN_TRAINING_SAMPLES = 100   # minimum windows required to start training

# --- Collection schedule ---
COLLECT_INTERVAL_SECONDS = 60

# --- Web Dashboard ---
DASHBOARD_HOST = "127.0.0.1"  # nur lokal, nginx proxied von außen
DASHBOARD_PORT = 5000
DASHBOARD_REFRESH_SECONDS = 30
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "iota2024")
DASHBOARD_SECRET_KEY = os.environ.get("DASHBOARD_SECRET_KEY", "change-this-secret-key-in-production")

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
