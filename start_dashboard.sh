#!/bin/bash
# start_dashboard.sh — Startet das Web-Dashboard in einer tmux-Session

LOGFILE=/home/user/Trader/logs/dashboard-start.log
TRADER_DIR=/home/user/Trader
VENV="$TRADER_DIR/venv/bin/activate"
TMUX_SESSION="iota-dashboard"

echo "=== $(date) Start Dashboard ===" >> "$LOGFILE"

export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:/home/user/.local/bin

cd "$TRADER_DIR" || { echo "Ordner fehlt: $TRADER_DIR" >> "$LOGFILE"; exit 1; }

if [ ! -f "$VENV" ]; then
    echo "Virtuelle Umgebung nicht gefunden." >> "$LOGFILE"
    exit 1
fi

echo "Stoppe alte Dashboard-Prozesse..." >> "$LOGFILE"
pkill -f "web_dashboard.py" 2>/dev/null
sleep 1

echo "Beende alte tmux Session '$TMUX_SESSION'..." >> "$LOGFILE"
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
sleep 1

echo "Starte neue tmux Session '$TMUX_SESSION'..." >> "$LOGFILE"
tmux new-session -d -s "$TMUX_SESSION" \
    "source $VENV && cd $TRADER_DIR && python web_dashboard.py 2>&1 | tee -a logs/dashboard.log"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "=== $(date) Dashboard erfolgreich gestartet ===" >> "$LOGFILE"
    echo ""
    echo "Dashboard läuft in tmux-Session '$TMUX_SESSION'."
    echo ""
    echo "  Lokal erreichbar:       http://127.0.0.1:5000"
    echo "  Live-Ausgabe anzeigen:  tmux attach -t $TMUX_SESSION"
    echo "  Session verlassen:      Ctrl+B, dann D"
    echo "  Logs verfolgen:         tail -f $TRADER_DIR/logs/dashboard.log"
    echo "  Stoppen:                bash stop_dashboard.sh"
    echo ""
    echo "  nginx-Konfiguration einrichten:"
    echo "  sudo cp $TRADER_DIR/nginx-dashboard.conf /etc/nginx/sites-available/trader"
    echo "  sudo ln -s /etc/nginx/sites-available/trader /etc/nginx/sites-enabled/"
    echo "  sudo nginx -t && sudo systemctl reload nginx"
    echo ""
else
    echo "=== $(date) FEHLER: tmux-Session konnte nicht gestartet werden ===" >> "$LOGFILE"
    exit 1
fi
