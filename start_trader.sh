#!/bin/bash
# start_trader.sh — Startet den Paper-Trading-Loop in einer tmux-Session

TRADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$TRADER_DIR/logs/trader-start.log"
VENV="$TRADER_DIR/venv/bin/activate"
TMUX_SESSION="iota-trader"

mkdir -p "$TRADER_DIR/logs"
echo "=== $(date) Start Trader ===" >> "$LOGFILE"

export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:"$HOME/.local/bin"

cd "$TRADER_DIR" || { echo "Ordner fehlt: $TRADER_DIR" >> "$LOGFILE"; exit 1; }

if [ ! -f "$VENV" ]; then
    echo "Virtuelle Umgebung nicht gefunden: $VENV" >> "$LOGFILE"
    exit 1
fi

pkill -f "main.py trade" 2>/dev/null
sleep 1
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
sleep 1

tmux new-session -d -s "$TMUX_SESSION" \
    "source $VENV && cd $TRADER_DIR && python main.py trade 2>&1 | tee -a logs/trader.log"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "=== $(date) Trader erfolgreich gestartet ===" >> "$LOGFILE"
    echo "Trader läuft in tmux-Session '$TMUX_SESSION'."
    echo "  Logs: tail -f $TRADER_DIR/logs/trader.log"
    echo "  Stoppen: bash $TRADER_DIR/stop_trader.sh"
else
    echo "=== $(date) FEHLER: tmux-Session konnte nicht gestartet werden ===" >> "$LOGFILE"
    exit 1
fi
