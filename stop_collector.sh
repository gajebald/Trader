#!/bin/bash
# stop_collector.sh — Stoppt den IOTA-Datensammler

LOGFILE=/home/user/Trader/logs/collector-start.log
TMUX_SESSION="iota-collector"

echo "=== $(date) Stop Script ===" >> "$LOGFILE"

echo "Stoppe Collector-Prozesse..." >> "$LOGFILE"
pkill -f "main.py collect" 2>/dev/null

echo "Beende tmux Session '$TMUX_SESSION'..." >> "$LOGFILE"
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null

echo "=== $(date) Collector gestoppt ===" >> "$LOGFILE"
echo "Collector gestoppt."
