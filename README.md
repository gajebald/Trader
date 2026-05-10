# IOTA Trading Bot

A Python paper-trading bot for IOTA/USD using Bitfinex public market data, technical indicators, and a local LLM (Ollama) for decision support.

**No real trades. No API keys. Paper-trading only.**

---

## Prerequisites

- Ubuntu 22.04+ (or any modern Linux)
- Python 3.11+
- Internet access (for Bitfinex API)
- Ollama (optional — bot falls back to HOLD when unavailable)

---

## Installation

```bash
# 1. Clone the repo and enter the project directory
cd ~/Trader

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Ollama Setup (Local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (pick one)
ollama pull llama3
# or: ollama pull mistral
# or: ollama pull qwen

# Start the Ollama server (runs on http://localhost:11434)
ollama serve
```

> If Ollama is not running, the bot defaults to HOLD for all LLM decisions and logs a warning.

---

## Configuration

Edit `config.py` to adjust any defaults:

| Setting | Default | Description |
|---|---|---|
| `SYMBOL` | `tIOTUSD` | Trading pair |
| `OLLAMA_MODEL` | `llama3` | LLM model name |
| `STARTING_CAPITAL` | `1000.0` | Paper trading capital (USD) |
| `STOP_LOSS_PCT` | `0.03` | Stop-loss threshold (3%) |
| `TAKE_PROFIT_PCT` | `0.05` | Take-profit threshold (5%) |
| `TRADE_FEE_PCT` | `0.002` | Simulated fee (0.2%) |
| `LLM_CONFIDENCE_THRESHOLD` | `0.6` | Min LLM confidence to act |
| `COLLECT_INTERVAL_SECONDS` | `60` | Data collection interval |

---

## Usage

All commands are run from the project directory with the virtual environment active:

```bash
source venv/bin/activate
```

### Collect live data

```bash
python main.py collect
```

Runs continuously. Fetches ticker + 1m/5m/1h candles every 60 seconds and stores them in `trader.db`. Press `Ctrl+C` to stop.

To run in the background on a server:

```bash
nohup python main.py collect > logs/collect.log 2>&1 &
```

### Run one paper trading step

```bash
python main.py paper
```

Evaluates the current strategy (indicators + LLM) and executes a virtual BUY, SELL, or HOLD. Prints the portfolio status after each run.

> Requires at least 51 candles in the database. Run `collect` for a few minutes first.

### Show portfolio status

```bash
python main.py status
```

Displays current cash, IOTA holdings, portfolio value, and PnL.

### Run a backtest

```bash
python main.py backtest
python main.py backtest --timeframe 5m
python main.py backtest --timeframe 1m
```

Replays the indicator-based strategy on all stored historical candles and reports performance metrics.

> The backtest uses only technical indicators (no LLM) for deterministic, repeatable results.

---

## How It Works

```
collect  →  Bitfinex API  →  SQLite (trader.db)
                                    |
paper    →  indicators.py  ←────────+
         →  llm_advisor.py  (Ollama REST)
         →  strategy.py  (rules engine)
         →  paper_trader.py  (virtual execution)
                                    |
backtest →  indicators.py  ←────────+
         →  backtester.py  (indicator-only replay)
```

**Strategy rules:**
- No BUY when RSI > 70 (overbought)
- No SELL when RSI < 30 (oversold), unless stop-loss triggers
- Stop-loss: 3% below entry price (always overrides RSI guard)
- Take-profit: 5% above entry price
- LLM must return confidence ≥ 0.6 to act on its signal
- Maximum 1 open virtual position at a time

---

## Project Structure

```
Trader/
├── main.py            CLI entry point
├── config.py          All constants and tunables
├── database.py        SQLite setup and queries
├── data_collector.py  Bitfinex API data fetching
├── indicators.py      SMA, EMA, RSI, MACD, Volatility
├── llm_advisor.py     Ollama LLM integration
├── strategy.py        Signal combination and rules
├── paper_trader.py    Virtual trade execution
├── backtester.py      Historical strategy replay
├── requirements.txt   Python dependencies
└── trader.db          SQLite database (created on first run)
```

---

## Disclaimer

This project is for **educational and research purposes only**. It does not execute real trades, does not use real money, and makes no financial recommendations. Past simulated performance does not guarantee future results.
