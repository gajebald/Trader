#!/bin/bash
# start_collector.sh — Startet den IOTA-Datensammler in einer tmux-Session

LOGFILE=/home/user/Trader/logs/collector-start.log
TRADER_DIR=/home/user/Trader
VENV="$TRADER_DIR/venv/bin/activate"
TMUX_SESSION="iota-collector"

echo "=== $(date) Start Script ===" >> "$LOGFILE"

export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:/home/user/.local/bin

# Projektverzeichnis prüfen
cd "$TRADER_DIR" || { echo "Ordner fehlt: $TRADER_DIR" >> "$LOGFILE"; exit 1; }

# Virtuelle Umgebung prüfen
if [ ! -f "$VENV" ]; then
    echo "Virtuelle Umgebung nicht gefunden: $VENV" >> "$LOGFILE"
    echo "Bitte zuerst 'bash install.sh' ausführen." >> "$LOGFILE"
    exit 1
fi

echo "Stoppe alte Collector-Prozesse..." >> "$LOGFILE"
pkill -f "main.py collect" 2>/dev/null
sleep 2

echo "Beende alte tmux Session '$TMUX_SESSION'..." >> "$LOGFILE"
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
sleep 1

echo "Starte neue tmux Session '$TMUX_SESSION'..." >> "$LOGFILE"
tmux new-session -d -s "$TMUX_SESSION" \
    "source $VENV && cd $TRADER_DIR && python main.py collect 2>&1 | tee -a logs/collector.log"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "=== $(date) Collector erfolgreich gestartet ===" >> "$LOGFILE"
    echo ""
    echo "Collector läuft in tmux-Session '$TMUX_SESSION'."
    echo ""
    echo "  Live-Ausgabe anzeigen:  tmux attach -t $TMUX_SESSION"
    echo "  Session verlassen:      Ctrl+B, dann D"
    echo "  Logs verfolgen:         tail -f $TRADER_DIR/logs/collector.log"
    echo "  Stoppen:                bash stop_collector.sh"
    echo ""
else
    echo "=== $(date) FEHLER: tmux-Session konnte nicht gestartet werden ===" >> "$LOGFILE"
    exit 1
fi
