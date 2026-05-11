#!/bin/bash
# start_training.sh — Startet das LSTM-Training in einer tmux-Session
# Verwendung: bash start_training.sh [timeframe]   (default: 1h)

TRADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$TRADER_DIR/logs/training-start.log"
VENV="$TRADER_DIR/venv/bin/activate"
TMUX_SESSION="iota-training"
TIMEFRAME="${1:-1h}"

mkdir -p "$TRADER_DIR/logs"
echo "=== $(date) Start Training ($TIMEFRAME) ===" >> "$LOGFILE"

export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:"$HOME/.local/bin"

cd "$TRADER_DIR" || { echo "Ordner fehlt: $TRADER_DIR" >> "$LOGFILE"; exit 1; }

if [ ! -f "$VENV" ]; then
    echo "Virtuelle Umgebung nicht gefunden: $VENV" >> "$LOGFILE"
    exit 1
fi

# Falls noch ein altes Training läuft, abbrechen
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Training läuft bereits (Session '$TMUX_SESSION' aktiv)." | tee -a "$LOGFILE"
    exit 0
fi

tmux new-session -d -s "$TMUX_SESSION" \
    "source $VENV && cd $TRADER_DIR && python main.py train --timeframe $TIMEFRAME 2>&1 | tee -a logs/training.log; echo '=== Training beendet ===' >> logs/training.log"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "=== $(date) Training gestartet (Zeitrahmen: $TIMEFRAME) ===" >> "$LOGFILE"
    echo "Training läuft in tmux-Session '$TMUX_SESSION'."
    echo "  Logs: tail -f $TRADER_DIR/logs/training.log"
else
    echo "=== $(date) FEHLER: tmux-Session konnte nicht gestartet werden ===" >> "$LOGFILE"
    exit 1
fi
