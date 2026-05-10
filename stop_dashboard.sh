#!/bin/bash
# stop_dashboard.sh — Stoppt das Web-Dashboard

LOGFILE=/home/user/Trader/logs/dashboard-start.log
TMUX_SESSION="iota-dashboard"

echo "=== $(date) Stop Dashboard ===" >> "$LOGFILE"

pkill -f "web_dashboard.py" 2>/dev/null
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null

echo "=== $(date) Dashboard gestoppt ===" >> "$LOGFILE"
echo "Dashboard gestoppt."
