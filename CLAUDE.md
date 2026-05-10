# CLAUDE.md — Projektdokumentation für Claude Code

Dieses Dokument beschreibt die Architektur, Konventionen und Entwicklungsregeln für den IOTA-Trading-Bot. Es wird von Claude Code beim Arbeiten in diesem Repository automatisch gelesen.

---

## Projektübersicht

Python-Paper-Trading-Bot für IOTA/USD (Bitfinex). Kein Echtgeld-Trading. Keine API-Keys. Die Entscheidungslogik basiert auf technischen Indikatoren kombiniert mit einem selbst trainierten Keras-LSTM-Modell.

**Branch für Entwicklung:** `claude/iota-trading-bot-Xo4mz`

---

## Technischer Stack

- **Sprache:** Python 3.11+
- **Datenbank:** SQLite (WAL-Modus, via `sqlite3` stdlib)
- **Marktdaten:** Bitfinex v2 Public API (kein Auth erforderlich)
- **Indikatoren:** pandas (SMA, EMA, RSI Wilder-Methode, MACD, Volatilität)
- **ML-Modell:** TensorFlow/Keras — LSTM-Klassifikator
- **Scheduler:** `schedule`-Bibliothek
- **CLI:** `argparse` mit Subcommands

---

## Modulstruktur und Verantwortlichkeiten

```
config.py          Alle Konstanten. Keine Funktionen.
database.py        Einzige SQLite-Schnittstelle. Alle anderen Module importieren von hier.
data_collector.py  Bitfinex-API → database. Läuft dauerhaft als Loop.
indicators.py      Reine Funktionen auf DataFrames. Kein I/O.
keras_advisor.py   Lädt trainiertes Modell, gibt prediction dict zurück.
model_trainer.py   Trainingspipeline: DB → Features → Labels → LSTM → Datei.
strategy.py        Kombiniert Indikatoren + Modell-Ausgabe → finale Entscheidung.
paper_trader.py    Virtuelle Handelsausführung. Liest/schreibt trades-Tabelle.
backtester.py      Historische Simulation. Nur Indikatoren, kein Modell.
main.py            CLI-Einstiegspunkt. Keine Business-Logik.
```

**Datenfluss (eine Richtung, keine Zyklen):**
```
database.py ← data_collector, indicators (liest), paper_trader, backtester
strategy.py ← indicators, keras_advisor, database
paper_trader ← strategy, database
backtester  ← indicators, database
main.py     ← alle Module (nur via lokale Imports in Funktionen)
```

---

## Datenbankschema

**`ticker_data`**
```sql
timestamp INTEGER PRIMARY KEY, symbol TEXT, bid REAL, ask REAL,
last_price REAL, volume REAL, high REAL, low REAL, spread REAL, source TEXT
```

**`candle_data`**
```sql
id INTEGER PRIMARY KEY AUTOINCREMENT,
timestamp INTEGER, symbol TEXT, timeframe TEXT,
open REAL, close REAL, high REAL, low REAL, volume REAL,
UNIQUE(timestamp, symbol, timeframe)
```

**`trades`**
```sql
id INTEGER PRIMARY KEY AUTOINCREMENT,
timestamp INTEGER, action TEXT, price REAL, quantity REAL,
fee REAL, pnl REAL, portfolio_value REAL, reason TEXT
```

Offene Position: abgeleitet via SQL (letzter BUY nach letztem SELL) — keine separate State-Tabelle.

---

## LSTM-Modell

### Features (11, in `indicators.FEATURE_COLUMNS`, Reihenfolge fix)

```python
["rsi_norm", "price_vs_sma20", "price_vs_sma50", "sma20_vs_sma50",
 "price_vs_ema20", "macd_norm", "macd_signal_norm", "macd_hist_norm",
 "volatility_norm", "pct_change_clipped", "high_low_ratio"]
```

Alle Features sind preisnormiert (Verhältnisse), sodass das Modell über verschiedene Preisniveaus generalisiert.

### Label-Generierung (Look-ahead, in `model_trainer.py`)

```
future_return = (close[i + LOOKAHEAD_BARS] - close[i]) / close[i]
> +LABEL_THRESHOLD_PCT  →  BUY  (class 1)
< -LABEL_THRESHOLD_PCT  →  SELL (class 2)
sonst                   →  HOLD (class 0)
```

### Architektur

```
Input(LOOKBACK_STEPS=20, features=11)
→ LSTM(64, return_sequences=True) → Dropout(0.2)
→ LSTM(32) → Dropout(0.2)
→ Dense(16, relu)
→ Dense(3, softmax)   # HOLD=0, BUY=1, SELL=2
```

Loss: `sparse_categorical_crossentropy` + Class-Weighting (HOLD-Dominanz ausgleichen)
Early Stopping: `patience=8` auf `val_loss`

### Feature-Konsistenz (kritisch)

`prepare_model_features()` in `indicators.py` **muss** identisch zwischen Training (`model_trainer.py`) und Inferenz (`keras_advisor.py`) sein. Beide importieren `FEATURE_COLUMNS` und `prepare_model_features` aus `indicators.py`. Niemals Feature-Logik duplizieren.

---

## Strategie-Regeln (`strategy.py`)

Reihenfolge der Regelauswertung in `apply_rules()`:

1. **Wenn Position offen:**
   - Stop-Loss ≤ entry × (1 - 0.03) → **SELL** (überschreibt alle Guards)
   - Take-Profit ≥ entry × (1 + 0.05) → **SELL**
   - RSI > 70 UND Modell=SELL UND Konfidenz ≥ 0.6 → **SELL**
   - Sonst → **HOLD**

2. **Wenn keine Position:**
   - RSI > 70 → **HOLD** (kein Entry bei Überkauf)
   - Modell=BUY UND Konfidenz ≥ 0.6 → **BUY**
   - Sonst → **HOLD**

---

## Wichtige Implementierungsdetails (Fallstricke)

### Bitfinex API
- Candle-Feldformat: `[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]` — CLOSE ist Index 2, HIGH Index 3 (nicht Standard-OHLCV)
- Candles absteigend sortiert (neueste zuerst) — `?sort=1` Query-Parameter für aufsteigende Reihenfolge verwenden

### Datenbank
- `PRAGMA journal_mode=WAL` in `setup_database()` — verhindert Reader/Writer-Blockierung zwischen Collect-Loop und Paper-Trader
- `INSERT OR IGNORE` für Candles (nie `INSERT OR REPLACE`) — schützt saubere historische Daten vor Überschreiben

### Indikatoren
- RSI: Wilder-Glättung via `ewm(com=period-1, adjust=False)` auf Gains/Losses — nicht `ewm(span=period)`
- Alle numpy-Werte in `get_latest_signals()` mit `float()` casten — sonst JSON-Fehler

### Paper-Trader
- Gebühreninklusiver Kauf: `qty = (cash - fee) / price`, nicht `qty = cash / price` dann Gebühr abziehen
- Portfolio-Wert immer durch vollständiges Replay aller Trades berechnen — kein gecachter State

### Keras-Modell
- Modell wird lazy geladen (einmalig beim ersten Aufruf von `get_advice()`)
- Kein Modell vorhanden → HOLD mit `reason="model_not_ready"` zurückgeben, nie crashen
- Input-Shape: `(1, LOOKBACK_STEPS, len(FEATURE_COLUMNS))` für Batch-Dimension

### Backtester
- Kein LLM/Modell im Backtest — nur Indikatoren, für Reproduzierbarkeit
- Ersten `SMA_LONG` (50) Rows droppen bevor Simulation startet (NaN in Indikatoren)

---

## CLI-Befehle

```bash
python main.py collect              # Dauerhafter Datensammel-Loop
python main.py train                # LSTM auf 1h-Candles trainieren
python main.py train --timeframe 5m # LSTM auf 5m-Candles trainieren
python main.py paper                # Eine Paper-Trading-Iteration
python main.py backtest             # Backtest auf 1h-Candles
python main.py backtest --timeframe 5m
python main.py status               # Portfolio-Status
```

---

## Code-Konventionen

- Keine unnötigen Kommentare — nur wenn das WARUM nicht offensichtlich ist
- Keine abstrakten Klassen oder Frameworks für v1 — direkte, lesbare Funktionen
- Error-Handling nur an Systemgrenzen (API-Calls, DB, Modell-Load)
- Logging via `logging.getLogger(__name__)` in jedem Modul
- `setup_database()` wird in jedem CLI-Befehl aufgerufen — ist idempotent
- Kein `import *`, alle Imports explizit

## Entwicklungsregeln

- Kein Echtgeld-Trading einbauen
- Keine Bitfinex-Auth-API-Keys verwenden
- Neue Features erst in Paper-Trading testen, dann Backtest
- Bei Änderungen an `FEATURE_COLUMNS` oder `prepare_model_features()`: Modell muss neu trainiert werden
- Bei Schema-Änderungen an der DB: `trader.db` löschen und neu erstellen (keine Migrationen in v1)
