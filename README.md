# IOTA Trading Bot

Ein Python-Paper-Trading-Bot für IOTA/USD auf Basis von Bitfinex-Marktdaten, technischen Indikatoren und einem selbst trainierten Keras-LSTM-Modell zur Entscheidungsunterstützung.

**Kein Echtgeld-Trading. Keine API-Keys. Nur Paper-Trading.**

---

## Voraussetzungen

- Ubuntu 22.04+ (oder ein aktuelles Linux-System)
- Python 3.11+
- Git
- Internetzugang (für die Bitfinex-API)
- Ca. 500 MB Speicherplatz (für TensorFlow) + ca. 1 GB für die Datenbank

---

## Installation auf dem Server

### Schritt 1 — Systempakete installieren

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
```

Python-Version prüfen (muss 3.11+):

```bash
python3 --version
```

Falls die Version zu alt ist (Ubuntu 20.04):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-pip
```

### Schritt 2 — Repository klonen

```bash
cd ~
git clone https://github.com/gajebald/Trader.git
cd Trader
```

Bei späteren Updates:

```bash
cd ~/Trader
git pull origin main
```

### Schritt 3 — Virtuelle Umgebung erstellen

```bash
python3 -m venv venv
source venv/bin/activate
```

Die virtuelle Umgebung muss bei jeder neuen SSH-Sitzung erneut aktiviert werden:

```bash
source ~/Trader/venv/bin/activate
```

### Schritt 4 — Abhängigkeiten installieren

Standard (mit GPU-Unterstützung):

```bash
pip install -r requirements.txt
```

Auf reinen CPU-Servern (ohne Grafikkarte, schlanker und schneller):

```bash
pip install requests>=2.31.0 pandas>=2.0.0 schedule>=1.2.0 numpy>=1.24.0
pip install tensorflow-cpu>=2.13.0
```

Installation prüfen:

```bash
python3 -c "import tensorflow as tf; print('TF:', tf.__version__)"
python3 -c "import pandas as pd; print('pandas:', pd.__version__)"
```

### Schritt 5 — Verzeichnisse anlegen

```bash
mkdir -p ~/Trader/logs
```

---

## Konfiguration

Alle Einstellungen befinden sich in `config.py`:

| Einstellung | Standard | Beschreibung |
|---|---|---|
| `SYMBOL` | `tIOTUSD` | Handelspaar |
| `STARTING_CAPITAL` | `1000.0` | Startkapital in USD |
| `STOP_LOSS_PCT` | `0.03` | Stop-Loss-Schwelle (3 %) |
| `TAKE_PROFIT_PCT` | `0.05` | Take-Profit-Schwelle (5 %) |
| `TRADE_FEE_PCT` | `0.002` | Simulierte Handelsgebühr (0,2 %) |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.6` | Mindest-Konfidenz des Modells zum Handeln |
| `COLLECT_INTERVAL_SECONDS` | `60` | Datensammlungsintervall in Sekunden |
| `LOOKBACK_STEPS` | `20` | Anzahl der Bars pro LSTM-Eingabefenster |
| `LOOKAHEAD_BARS` | `10` | Bars in die Zukunft für Label-Generierung |
| `LABEL_THRESHOLD_PCT` | `0.015` | Mindestbewegung (1,5 %) für BUY/SELL-Label |
| `TRAINING_EPOCHS` | `50` | Maximale Trainings-Epochen |
| `MODEL_PATH` | `models/trading_model.keras` | Speicherpfad des trainierten Modells |

---

## Typischer Workflow

```
1. collect   → Marktdaten sammeln (Stunden bis Tage)
2. train     → LSTM-Modell auf den gesammelten Daten trainieren
3. paper     → Paper-Trading mit dem trainierten Modell
4. backtest  → Historische Auswertung der Strategie
5. status    → Aktuellen Portfoliostand anzeigen
```

---

## Befehle

Alle Befehle werden im Projektverzeichnis mit aktivierter virtueller Umgebung ausgeführt:

```bash
source venv/bin/activate
```

### 1. Daten sammeln

```bash
python main.py collect
```

Läuft dauerhaft. Ruft alle 60 Sekunden Ticker- und Candle-Daten (1m/5m/1h) von der Bitfinex v2 Public API ab und speichert sie in `trader.db`. Mit `Ctrl+C` beenden.

Im Hintergrund auf einem Server:

```bash
mkdir -p logs
nohup python main.py collect > logs/collect.log 2>&1 &
```

### 2. Modell trainieren

```bash
python main.py train
python main.py train --timeframe 5m   # alternativ 1m oder 5m
```

Trainiert das LSTM-Modell auf den gespeicherten historischen Candles. Das Modell wird unter `models/trading_model.keras` gespeichert.

**Mindestanforderung an Daten (nach Indikator-Warmup):**

| Timeframe | Benötigte Sammelzeit |
|---|---|
| `1h` | ca. 11 Tage (270+ Stunden) |
| `5m` | ca. 17 Stunden |
| `1m` | ca. 3,5 Stunden |

> Das Training startet erst, wenn mindestens 100 saubere Trainings-Fenster vorhanden sind. Ohne ausreichend Daten gibt das Training eine klare Fehlermeldung aus.

### 3. Paper-Trading

```bash
python main.py paper
```

Führt eine virtuelle Handelsentscheidung durch (BUY / SELL / HOLD) und gibt den aktuellen Portfoliostand aus. Erfordert ein trainiertes Modell. Ohne Modell wird HOLD zurückgegeben und eine Warnung geloggt.

### 4. Portfolio-Status anzeigen

```bash
python main.py status
```

Zeigt Kassenstand, IOTA-Bestand, Portfoliowert und Gewinn/Verlust.

### 5. Backtest

```bash
python main.py backtest
python main.py backtest --timeframe 5m
python main.py backtest --timeframe 1m
```

Simuliert die Strategie auf allen gespeicherten historischen Candles und gibt Performance-Kennzahlen aus. Der Backtest verwendet **nur technische Indikatoren** (kein Modell), um deterministische und reproduzierbare Ergebnisse zu gewährleisten.

---

## Wie es funktioniert

```
collect  →  Bitfinex API  →  SQLite (trader.db)
                                    │
train    →  indicators.py  ◄────────┤
         →  model_trainer.py        │   (LSTM trainieren + speichern)
                                    │
paper    →  indicators.py  ◄────────┤
         →  keras_advisor.py        │   (LSTM-Inferenz)
         →  strategy.py             │   (Regelwerk)
         →  paper_trader.py         │   (virtuelle Ausführung)
                                    │
backtest →  indicators.py  ◄────────┘
         →  backtester.py           (nur Indikatoren, kein Modell)
```

### LSTM-Modell

Das Modell lernt aus historischen Marktbewegungen:

- **Eingabe:** Sequenz der letzten 20 Bars mit 11 normalisierten Features
- **Features:** RSI (normiert), Preis zu SMA20/SMA50/EMA20, MACD (normiert), Volatilität, prozentuale Preisänderung, High-Low-Verhältnis
- **Label-Generierung:** Look-ahead über 10 Bars — steigt der Preis um >1,5 %, ist das Label BUY; fällt er um >1,5 %, SELL; sonst HOLD
- **Architektur:** `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(3, softmax)`
- **Ausgabe:** Wahrscheinlichkeitsverteilung über BUY / SELL / HOLD

### Strategie-Regeln (zusätzlich zum Modell)

- Kein BUY, wenn RSI > 70 (überkauft)
- Kein SELL, wenn RSI < 30 (überverkauft) — außer Stop-Loss greift
- Stop-Loss: 3 % unter Einstiegspreis (überschreibt immer den RSI-Guard)
- Take-Profit: 5 % über Einstiegspreis
- Modell-Konfidenz muss ≥ 0,6 sein, damit ein Signal ausgeführt wird
- Maximal 1 offene virtuelle Position gleichzeitig

---

## Projektstruktur

```
Trader/
├── main.py              CLI-Einstiegspunkt
├── config.py            Alle Konstanten und Parameter
├── database.py          SQLite-Setup und Abfragen
├── data_collector.py    Bitfinex-API-Datenabruf
├── indicators.py        SMA, EMA, RSI, MACD, Volatilität + Feature-Engineering
├── keras_advisor.py     LSTM-Modell-Inferenz
├── model_trainer.py     LSTM-Training und Label-Generierung
├── strategy.py          Regelwerk (Indikatoren + Modell)
├── paper_trader.py      Virtuelle Handelsausführung
├── backtester.py        Historische Strategie-Simulation
├── requirements.txt     Python-Abhängigkeiten
├── models/              Gespeicherte Keras-Modelle (nicht im Git)
└── trader.db            SQLite-Datenbank (wird beim ersten Start erstellt)
```

---

## Disclaimer

Dieses Projekt dient ausschließlich **Bildungs- und Forschungszwecken**. Es führt keine echten Trades aus, verwendet kein echtes Geld und gibt keine Finanzempfehlungen. Vergangene Simulationsergebnisse sind kein Indikator für zukünftige Ergebnisse.
