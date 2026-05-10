#!/bin/bash
# stop_dashboard.sh — Stoppt das Web-Dashboard

TRADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$TRADER_DIR/logs/dashboard-start.log"
TMUX_SESSION="iota-dashboard"

mkdir -p "$TRADER_DIR/logs"
echo "=== $(date) Stop Dashboard ===" >> "$LOGFILE"

pkill -f "web_dashboard.py" 2>/dev/null
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null

echo "=== $(date) Dashboard gestoppt ===" >> "$LOGFILE"
echo "Dashboard gestoppt."
