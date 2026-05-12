#!/bin/bash
# stop_collector.sh — Stoppt den IOTA-Datensammler

TRADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$TRADER_DIR/logs/collector-start.log"
TMUX_SESSION="iota-collector"

mkdir -p "$TRADER_DIR/logs"
echo "=== $(date) Stop Script ===" >> "$LOGFILE"

pkill -f "main.py collect" 2>/dev/null
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null

echo "=== $(date) Collector gestoppt ===" >> "$LOGFILE"
echo "Collector gestoppt."
