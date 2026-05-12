#!/bin/bash
# start_history.sh — Lädt historische Candles von Bitfinex
# Verwendung: bash start_history.sh [timeframe] [days]

TRADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$TRADER_DIR/logs/history-start.log"
VENV="$TRADER_DIR/venv/bin/activate"
TMUX_SESSION="iota-history"
TIMEFRAME="${1:-1h}"
DAYS="${2:-365}"

mkdir -p "$TRADER_DIR/logs"
echo "=== $(date) Start History ($TIMEFRAME, ${DAYS}d) ===" >> "$LOGFILE"

export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:"$HOME/.local/bin"

cd "$TRADER_DIR" || { echo "Ordner fehlt: $TRADER_DIR" >> "$LOGFILE"; exit 1; }

if [ ! -f "$VENV" ]; then
    echo "Virtuelle Umgebung nicht gefunden: $VENV" >> "$LOGFILE"
    exit 1
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Download läuft bereits (Session '$TMUX_SESSION' aktiv)."
    exit 0
fi

tmux new-session -d -s "$TMUX_SESSION" \
    "source $VENV && cd $TRADER_DIR && python main.py history --timeframe $TIMEFRAME --days $DAYS 2>&1 | tee -a logs/history.log"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "=== $(date) History-Download gestartet ===" >> "$LOGFILE"
    echo "Download läuft in tmux-Session '$TMUX_SESSION'."
    echo "  Logs: tail -f $TRADER_DIR/logs/history.log"
else
    echo "=== $(date) FEHLER: Session konnte nicht gestartet werden ===" >> "$LOGFILE"
    exit 1
fi
