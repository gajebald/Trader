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
STOP_LOSS_PCT = 0.03      # 3% below entry price
TAKE_PROFIT_PCT = 0.05    # 5% above entry price
TRADE_FEE_PCT = 0.002     # 0.2% per trade
LLM_CONFIDENCE_THRESHOLD = 0.6

# --- Paper trading ---
STARTING_CAPITAL = 1000.0  # USD

# --- LLM (Ollama) ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 30  # seconds

# --- Collection schedule ---
COLLECT_INTERVAL_SECONDS = 60
