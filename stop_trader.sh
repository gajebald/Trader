#!/bin/bash
# stop_trader.sh — Stoppt den Paper-Trading-Loop

TRADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$TRADER_DIR/logs/trader-start.log"
TMUX_SESSION="iota-trader"

mkdir -p "$TRADER_DIR/logs"
echo "=== $(date) Stop Trader ===" >> "$LOGFILE"

pkill -f "main.py trade" 2>/dev/null
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null

echo "=== $(date) Trader gestoppt ===" >> "$LOGFILE"
echo "Trader gestoppt."
