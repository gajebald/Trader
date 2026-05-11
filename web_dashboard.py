"""
Web-Dashboard für den IOTA Trading Bot.
Zeigt Portfolio, Trades und aktuelle Marktsignale.
Läuft auf localhost, nginx proxied von außen.
"""
import os
import sys
import subprocess
import time as _time
import logging
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Flask, render_template_string, request, redirect, url_for, session, jsonify

import json

from config import (
    SYMBOL, STARTING_CAPITAL, DASHBOARD_HOST, DASHBOARD_PORT,
    DASHBOARD_REFRESH_SECONDS, MODEL_PATH, MODEL_METRICS_PATH,
    LOOKBACK_STEPS, LOOKAHEAD_BARS, LABEL_THRESHOLD_PCT, MIN_TRAINING_SAMPLES,
    DASHBOARD_PASSWORD, DASHBOARD_SECRET_KEY,
)
from database import get_all_trades, get_open_position, get_recent_candles, get_data_stats
from indicators import FEATURE_COLUMNS, calculate_all, get_latest_signals, prepare_model_features
from paper_trader import get_portfolio_status

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
PYTHON = sys.executable

app = Flask(__name__)
app.secret_key = DASHBOARD_SECRET_KEY


# -------------------------------------------------------
# Auth
# -------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# -------------------------------------------------------
# Login-Template
# -------------------------------------------------------
_LOGIN_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IOTA Trading Bot — Login</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .login-box {
      background: #1e2330;
      border: 1px solid #2d3748;
      border-radius: 14px;
      padding: 40px 36px;
      width: 100%;
      max-width: 360px;
      text-align: center;
    }
    h1 { font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }
    .subtitle { font-size: 0.8rem; color: #64748b; margin-bottom: 28px; }
    input[type=password] {
      width: 100%;
      padding: 11px 14px;
      background: #161b27;
      border: 1px solid #2d3748;
      border-radius: 8px;
      color: #e2e8f0;
      font-size: 0.95rem;
      outline: none;
      margin-bottom: 14px;
    }
    input[type=password]:focus { border-color: #38bdf8; }
    button {
      width: 100%;
      padding: 11px;
      background: #38bdf8;
      color: #0f1117;
      font-weight: 700;
      font-size: 0.95rem;
      border: none;
      border-radius: 8px;
      cursor: pointer;
    }
    button:hover { background: #7dd3fc; }
    .error { color: #f87171; font-size: 0.82rem; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="login-box">
    <h1>Admin Login</h1>
    <p class="subtitle">Bitte anmelden um fortzufahren</p>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="post">
      <input type="password" name="password" placeholder="Passwort" autofocus>
      <button type="submit">Anmelden</button>
    </form>
  </div>
</body>
</html>"""


# -------------------------------------------------------
# Haupt-Template
# -------------------------------------------------------
_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="{{ refresh }}">
  <title>IOTA Trading Bot</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 24px 16px;
    }
    .header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 28px; }
    h1 { font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .subtitle { color: #64748b; font-size: 0.85rem; }
    .subtitle .updated { color: #38bdf8; }
    .logout-btn {
      padding: 7px 16px;
      background: #1e2330;
      border: 1px solid #2d3748;
      border-radius: 8px;
      color: #94a3b8;
      font-size: 0.8rem;
      cursor: pointer;
      text-decoration: none;
      white-space: nowrap;
    }
    .logout-btn:hover { border-color: #f87171; color: #f87171; }

    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 28px; }
    .card {
      background: #1e2330;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 16px;
    }
    .card-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em; color: #64748b; margin-bottom: 6px; }
    .card-value { font-size: 1.35rem; font-weight: 700; }
    .card-value.green  { color: #4ade80; }
    .card-value.red    { color: #f87171; }
    .card-value.yellow { color: #facc15; }
    .card-value.blue   { color: #38bdf8; }
    .card-value.gray   { color: #94a3b8; }

    .section-title {
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: #64748b;
      margin-bottom: 12px;
    }
    .panel { background: #1e2330; border: 1px solid #2d3748; border-radius: 10px; padding: 18px; margin-bottom: 28px; }

    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { text-align: left; padding: 8px 10px; color: #64748b; font-weight: 600;
         border-bottom: 1px solid #2d3748; text-transform: uppercase; font-size: 0.72rem; letter-spacing: .06em; }
    td { padding: 9px 10px; border-bottom: 1px solid #1a2033; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #232a3b; }
    .buy  { color: #4ade80; font-weight: 700; }
    .sell { color: #f87171; font-weight: 700; }
    .hold { color: #94a3b8; }
    .pos  { color: #4ade80; }
    .neg  { color: #f87171; }

    .signals-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
    .sig { background: #161b27; border-radius: 8px; padding: 10px 14px; }
    .sig-name  { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
    .sig-value { font-size: 1rem; font-weight: 600; margin-top: 3px; }

    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .dot-green  { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
    .dot-red    { background: #f87171; }
    .dot-yellow { background: #facc15; box-shadow: 0 0 6px #facc15; }

    .empty { color: #475569; font-style: italic; text-align: center; padding: 20px; }
    footer { text-align: center; color: #334155; font-size: 0.75rem; margin-top: 12px; }

    .tf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
    .tf-card { background: #161b27; border-radius: 8px; padding: 12px 14px; }
    .tf-name { font-size: 0.75rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
    .tf-count { font-size: 1.4rem; font-weight: 700; color: #f8fafc; }
    .tf-label { font-size: 0.7rem; color: #64748b; margin-top: 2px; }
    .tf-range { font-size: 0.72rem; color: #475569; margin-top: 6px; border-top: 1px solid #1e2330; padding-top: 6px; }

    /* Steuerung */
    .ctrl-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
    .ctrl-card { background: #161b27; border-radius: 10px; padding: 16px 18px; }
    .ctrl-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em; color: #64748b; margin-bottom: 10px; }
    .ctrl-status { font-size: 0.9rem; font-weight: 600; min-height: 22px; margin-bottom: 14px; }
    .btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .btn {
      padding: 8px 16px;
      border: none;
      border-radius: 7px;
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
      transition: opacity .15s;
    }
    .btn:hover { opacity: .85; }
    .btn:disabled { opacity: .4; cursor: default; }
    .btn-green { background: #166534; color: #4ade80; border: 1px solid #166534; }
    .btn-red   { background: #7f1d1d; color: #f87171; border: 1px solid #7f1d1d; }
    .btn-blue  { background: #0c4a6e; color: #38bdf8; border: 1px solid #0c4a6e; }
    .ctrl-log {
      margin-top: 10px;
      font-size: 0.75rem;
      color: #64748b;
      font-family: monospace;
      background: #0f1117;
      border-radius: 6px;
      padding: 8px 10px;
      min-height: 36px;
      white-space: pre-wrap;
      word-break: break-all;
    }
  </style>
</head>
<body>

<div class="header">
  <div>
    <h1>📈 IOTA Trading Bot</h1>
    <p class="subtitle">
      Symbol: <strong>{{ symbol }}</strong> &nbsp;|&nbsp;
      Aktualisierung alle {{ refresh }}s &nbsp;|&nbsp;
      <span class="updated">{{ now }}</span>
    </p>
  </div>
  <a href="/logout" class="logout-btn">Abmelden</a>
</div>

<!-- Steuerung -->
<div class="section-title">Steuerung</div>
<div class="panel">
  <div class="ctrl-grid">

    <div class="ctrl-card">
      <div class="ctrl-title">Datensammler</div>
      <div class="ctrl-status" id="collector-status">
        <span class="status-dot dot-yellow"></span>Prüfe…
      </div>
      <div class="btn-row">
        <button class="btn btn-green" onclick="collectorAction('start')">▶ Starten</button>
        <button class="btn btn-red"   onclick="collectorAction('stop')">■ Stoppen</button>
      </div>
    </div>

    <div class="ctrl-card">
      <div class="ctrl-title">Trading-Loop</div>
      <div class="ctrl-status" id="trader-status">
        <span class="status-dot dot-yellow"></span>Prüfe…
      </div>
      <div class="btn-row">
        <button class="btn btn-green" onclick="traderAction('start')">▶ Starten</button>
        <button class="btn btn-red"   onclick="traderAction('stop')">■ Stoppen</button>
      </div>
    </div>

    <div class="ctrl-card">
      <div class="ctrl-title">Modell-Training</div>
      <div class="ctrl-status" id="training-status">
        <span class="status-dot dot-yellow"></span>Prüfe…
      </div>
      <div class="btn-row" style="align-items:center">
        <select id="training-tf" style="padding:7px 10px;background:#0f1117;border:1px solid #2d3748;border-radius:7px;color:#e2e8f0;font-size:0.82rem;">
          <option value="1h">1h</option>
          <option value="5m">5m</option>
          <option value="1m">1m</option>
        </select>
        <button class="btn btn-blue" id="training-btn" onclick="startTraining()">🧠 Training starten</button>
      </div>
      <div class="ctrl-log" id="training-log"></div>
    </div>

    <div class="ctrl-card">
      <div class="ctrl-title">Paper Trader (einmalig)</div>
      <div class="ctrl-status" id="paper-status" style="color:#64748b">—</div>
      <div class="btn-row">
        <button class="btn btn-blue" id="paper-btn" onclick="runPaper()">⚡ Iteration ausführen</button>
      </div>
      <div class="ctrl-log" id="paper-log"></div>
    </div>

  </div>
</div>

<!-- Portfolio-Karten -->
<div class="grid">
  <div class="card">
    <div class="card-label">Portfolio-Wert</div>
    <div class="card-value blue">${{ status.portfolio_value }}</div>
  </div>
  <div class="card">
    <div class="card-label">Startkapital</div>
    <div class="card-value gray">${{ "%.2f"|format(start_capital) }}</div>
  </div>
  <div class="card">
    <div class="card-label">Kassenstand</div>
    <div class="card-value">${{ status.cash }}</div>
  </div>
  <div class="card">
    <div class="card-label">IOTA-Bestand</div>
    <div class="card-value">{{ status.iota_holdings }} IOTA</div>
  </div>
  <div class="card">
    <div class="card-label">Realisierter PnL</div>
    <div class="card-value {{ 'green' if status.realized_pnl >= 0 else 'red' }}">
      {{ '+' if status.realized_pnl >= 0 else '' }}${{ status.realized_pnl }}
    </div>
  </div>
  <div class="card">
    <div class="card-label">Unrealisierter PnL</div>
    <div class="card-value {{ 'green' if status.unrealized_pnl >= 0 else 'red' }}">
      {{ '+' if status.unrealized_pnl >= 0 else '' }}${{ status.unrealized_pnl }}
    </div>
  </div>
  <div class="card">
    <div class="card-label">Aktueller Preis</div>
    <div class="card-value yellow">${{ status.current_price }}</div>
  </div>
  <div class="card">
    <div class="card-label">Trades gesamt</div>
    <div class="card-value">{{ status.trade_count }}</div>
  </div>
</div>

<!-- Position & Modell -->
<div class="grid" style="margin-bottom:28px">
  <div class="card">
    <div class="card-label">Offene Position</div>
    <div class="card-value {{ 'green' if position else 'gray' }}">
      {% if position %}
        <span class="status-dot dot-green"></span>OPEN @ ${{ "%.4f"|format(position.price) }}
      {% else %}
        <span class="status-dot dot-red"></span>Keine
      {% endif %}
    </div>
  </div>
  <div class="card">
    <div class="card-label">Keras-Modell</div>
    <div class="card-value {{ 'green' if model_ready else 'yellow' }}">
      {% if model_ready %}
        <span class="status-dot dot-green"></span>Bereit
      {% else %}
        <span class="status-dot dot-yellow"></span>Nicht trainiert
      {% endif %}
    </div>
  </div>
</div>

<!-- Gesammelte Daten -->
<div class="section-title">Gesammelte Daten</div>
<div class="panel">
  <div class="tf-grid">
    <div class="tf-card">
      <div class="tf-name">Ticker</div>
      <div class="tf-count">{{ stats.ticker_count }}</div>
      <div class="tf-label">Einträge gesamt</div>
      {% if stats.latest_ticker_ts %}
      <div class="tf-range">Letzter: {{ stats.latest_ticker_ts | ts }}</div>
      {% endif %}
    </div>
    {% for tf, d in stats.timeframes.items() %}
    <div class="tf-card">
      <div class="tf-name">{{ tf }} Candles</div>
      <div class="tf-count {{ 'green' if d.count >= 100 else ('yellow' if d.count > 0 else 'gray') }}">
        {{ d.count }}
      </div>
      <div class="tf-label">
        {% if d.count >= 100 %}ausreichend für Training
        {% elif d.count > 0 %}zu wenig für Training (min. 100)
        {% else %}noch keine Daten
        {% endif %}
      </div>
      {% if d.first_ts and d.last_ts %}
      <div class="tf-range">
        {{ d.first_ts | ts }} –<br>{{ d.last_ts | ts }}
      </div>
      {% endif %}
    </div>
    {% endfor %}
  </div>
</div>

<!-- Trainingsdaten -->
<div class="section-title">Trainingsdaten</div>
<div class="panel">
  <table>
    <thead>
      <tr>
        <th>Zeitrahmen</th>
        <th>Candles</th>
        <th>Saubere Zeilen</th>
        <th>Sequenzen</th>
        <th>HOLD</th>
        <th>BUY</th>
        <th>SELL</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {% for tf, d in training_stats.items() %}
      <tr>
        <td style="color:#38bdf8;font-weight:700">{{ tf }}</td>
        <td>{{ d.candles }}</td>
        <td>{{ d.clean_rows }}</td>
        <td>{{ d.sequences }}</td>
        <td class="hold">{{ d.hold }}</td>
        <td class="buy">{{ d.buy }}</td>
        <td class="sell">{{ d.sell }}</td>
        <td>
          {% if d.ready %}
            <span class="status-dot dot-green"></span><span style="color:#4ade80">Bereit</span>
          {% elif d.sequences > 0 %}
            <span class="status-dot dot-yellow"></span><span style="color:#facc15">Zu wenig (min. {{ min_samples }})</span>
          {% else %}
            <span class="status-dot dot-red"></span><span style="color:#f87171">Keine Daten</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  <div style="margin-top:12px;font-size:0.78rem;color:#475569">
    Lookback: <strong style="color:#94a3b8">{{ lookback_steps }} Bars</strong> &nbsp;|&nbsp;
    Lookahead: <strong style="color:#94a3b8">{{ lookahead_bars }} Bars</strong> &nbsp;|&nbsp;
    Schwelle: <strong style="color:#94a3b8">±{{ "%.1f"|format(label_threshold_pct * 100) }}%</strong>
    &nbsp;|&nbsp;
    Training starten:
    <code style="color:#38bdf8;background:#161b27;padding:2px 6px;border-radius:4px">python main.py train --timeframe 1h</code>
  </div>
</div>

<!-- Modell-Metriken -->
<div class="section-title">Modell-Metriken</div>
<div class="panel">
  {% if metrics %}
  <div class="grid" style="margin-bottom:16px">
    <div class="card">
      <div class="card-label">Val Loss</div>
      <div class="card-value {{ 'green' if metrics.val_loss < 1.0 else ('yellow' if metrics.val_loss < 1.2 else 'red') }}">
        {{ "%.4f"|format(metrics.val_loss) }}
      </div>
    </div>
    <div class="card">
      <div class="card-label">Val Accuracy</div>
      <div class="card-value {{ 'green' if metrics.val_accuracy >= 0.45 else ('yellow' if metrics.val_accuracy >= 0.35 else 'red') }}">
        {{ "%.1f"|format(metrics.val_accuracy * 100) }}%
      </div>
    </div>
    <div class="card">
      <div class="card-label">Epochen</div>
      <div class="card-value gray">{{ metrics.epochs_ran }}</div>
    </div>
    <div class="card">
      <div class="card-label">Zeitrahmen</div>
      <div class="card-value blue">{{ metrics.timeframe }}</div>
    </div>
    <div class="card">
      <div class="card-label">Train-Samples</div>
      <div class="card-value gray">{{ metrics.train_samples }}</div>
    </div>
    <div class="card">
      <div class="card-label">Val-Samples</div>
      <div class="card-value gray">{{ metrics.val_samples }}</div>
    </div>
  </div>
  {% if metrics.class_weights %}
  <div style="font-size:0.8rem;color:#64748b;margin-bottom:14px">
    Klassen-Gewichte: &nbsp;
    <span style="color:#94a3b8">HOLD={{ metrics.class_weights.HOLD }}</span> &nbsp;
    <span style="color:#4ade80">BUY={{ metrics.class_weights.BUY }}</span> &nbsp;
    <span style="color:#f87171">SELL={{ metrics.class_weights.SELL }}</span>
    &nbsp;|&nbsp; Trainiert: <span style="color:#38bdf8">{{ metrics.trained_at }}</span>
  </div>
  {% endif %}
  {{ loss_chart | safe }}
  {% else %}
  <div class="empty">Noch kein Modell trainiert — Training über den Button starten.</div>
  {% endif %}
</div>

<!-- Technische Signale -->
<div class="section-title">Technische Signale (5m)</div>
<div class="panel">
  {% if signals %}
  <div class="signals-grid">
    <div class="sig">
      <div class="sig-name">Preis</div>
      <div class="sig-value yellow">${{ "%.4f"|format(signals.last_price) if signals.last_price else '—' }}</div>
    </div>
    <div class="sig">
      <div class="sig-name">RSI 14</div>
      <div class="sig-value {{ 'red' if signals.rsi and signals.rsi > 70 else ('green' if signals.rsi and signals.rsi < 30 else '') }}">
        {{ "%.1f"|format(signals.rsi) if signals.rsi else '—' }}
      </div>
    </div>
    <div class="sig">
      <div class="sig-name">SMA 20</div>
      <div class="sig-value">${{ "%.4f"|format(signals.sma_20) if signals.sma_20 else '—' }}</div>
    </div>
    <div class="sig">
      <div class="sig-name">SMA 50</div>
      <div class="sig-value">${{ "%.4f"|format(signals.sma_50) if signals.sma_50 else '—' }}</div>
    </div>
    <div class="sig">
      <div class="sig-name">EMA 20</div>
      <div class="sig-value">${{ "%.4f"|format(signals.ema_20) if signals.ema_20 else '—' }}</div>
    </div>
    <div class="sig">
      <div class="sig-name">MACD</div>
      <div class="sig-value {{ 'green' if signals.macd and signals.macd > 0 else 'red' }}">
        {{ "%.5f"|format(signals.macd) if signals.macd else '—' }}
      </div>
    </div>
    <div class="sig">
      <div class="sig-name">MACD Signal</div>
      <div class="sig-value">{{ "%.5f"|format(signals.macd_signal) if signals.macd_signal else '—' }}</div>
    </div>
    <div class="sig">
      <div class="sig-name">Volatilität</div>
      <div class="sig-value">{{ "%.5f"|format(signals.volatility) if signals.volatility else '—' }}</div>
    </div>
    <div class="sig">
      <div class="sig-name">Änderung %</div>
      <div class="sig-value {{ 'green' if signals.pct_change and signals.pct_change > 0 else 'red' }}">
        {{ '%+.3f%%'|format(signals.pct_change * 100) if signals.pct_change else '—' }}
      </div>
    </div>
  </div>
  {% else %}
  <div class="empty">Noch keine Signaldaten — bitte zuerst den Datensammler starten.</div>
  {% endif %}
</div>

<!-- Trade-Historie -->
<div class="section-title">Trade-Historie (letzte 20)</div>
<div class="panel">
  {% if trades %}
  <table>
    <thead>
      <tr>
        <th>Zeit</th>
        <th>Aktion</th>
        <th>Preis</th>
        <th>Menge</th>
        <th>Gebühr</th>
        <th>PnL</th>
        <th>Portfolio</th>
        <th>Grund</th>
      </tr>
    </thead>
    <tbody>
      {% for t in trades %}
      <tr>
        <td>{{ t.time }}</td>
        <td class="{{ t.action|lower }}">{{ t.action }}</td>
        <td>${{ "%.4f"|format(t.price) }}</td>
        <td>{{ "%.4f"|format(t.quantity) }}</td>
        <td>${{ "%.4f"|format(t.fee) }}</td>
        <td class="{{ 'pos' if t.pnl >= 0 else 'neg' }}">
          {{ '+' if t.pnl >= 0 else '' }}${{ "%.4f"|format(t.pnl) }}
        </td>
        <td>${{ "%.2f"|format(t.portfolio_value) }}</td>
        <td style="color:#64748b;font-size:0.78rem">{{ t.reason or '—' }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty">Noch keine Trades vorhanden.</div>
  {% endif %}
</div>

<footer>IOTA Trading Bot &mdash; Paper Trading Only &mdash; Kein Echtgeld</footer>

<script>
function _post(url) {
  return fetch(url, {method: 'POST', headers: {'X-Requested-With': 'XMLHttpRequest'}});
}

// ---- Collector status polling ----
function updateCollectorStatus() {
  fetch('/api/collector/status')
    .then(r => r.json())
    .then(data => {
      const el = document.getElementById('collector-status');
      if (data.running) {
        el.innerHTML = '<span class="status-dot dot-green"></span><span style="color:#4ade80">Läuft</span>';
      } else {
        el.innerHTML = '<span class="status-dot dot-red"></span><span style="color:#f87171">Gestoppt</span>';
      }
    })
    .catch(() => {
      document.getElementById('collector-status').innerHTML =
        '<span class="status-dot dot-yellow"></span><span style="color:#facc15">Unbekannt</span>';
    });
}

function collectorAction(action) {
  _post('/api/collector/' + action)
    .then(() => { setTimeout(updateCollectorStatus, 1800); });
}

// ---- Trader status polling ----
function updateTraderStatus() {
  fetch('/api/trader/status')
    .then(r => r.json())
    .then(data => {
      const el = document.getElementById('trader-status');
      if (data.running) {
        el.innerHTML = '<span class="status-dot dot-green"></span><span style="color:#4ade80">Läuft</span>';
      } else {
        el.innerHTML = '<span class="status-dot dot-red"></span><span style="color:#f87171">Gestoppt</span>';
      }
    })
    .catch(() => {
      document.getElementById('trader-status').innerHTML =
        '<span class="status-dot dot-yellow"></span><span style="color:#facc15">Unbekannt</span>';
    });
}

function traderAction(action) {
  _post('/api/trader/' + action)
    .then(() => { setTimeout(updateTraderStatus, 1800); });
}

// ---- Training ----
function updateTrainingStatus() {
  fetch('/api/training/status')
    .then(r => r.json())
    .then(data => {
      const el = document.getElementById('training-status');
      const btn = document.getElementById('training-btn');
      if (data.running) {
        el.innerHTML = '<span class="status-dot dot-yellow"></span><span style="color:#facc15">Läuft…</span>';
        btn.disabled = true;
      } else if (data.model_ready) {
        el.innerHTML = '<span class="status-dot dot-green"></span><span style="color:#4ade80">Modell bereit</span>';
        btn.disabled = false;
      } else {
        el.innerHTML = '<span class="status-dot dot-red"></span><span style="color:#f87171">Kein Modell</span>';
        btn.disabled = false;
      }
    });
}

function startTraining() {
  const tf = document.getElementById('training-tf').value;
  const logEl = document.getElementById('training-log');
  logEl.textContent = 'Training wird gestartet (' + tf + ')…';
  _post('/api/training/start?timeframe=' + tf)
    .then(r => r.json())
    .then(data => {
      logEl.textContent = data.message || '';
      setTimeout(updateTrainingStatus, 2000);
      // Weiter pollen bis Training fertig
      const poll = setInterval(() => {
        fetch('/api/training/status').then(r => r.json()).then(d => {
          if (!d.running) {
            clearInterval(poll);
            updateTrainingStatus();
            logEl.textContent = d.model_ready ? '✓ Training abgeschlossen — Modell gespeichert.' : '✗ Training beendet (Modell nicht gefunden).';
          }
        });
      }, 5000);
    });
}

// ---- Paper trader ----
function runPaper() {
  const btn = document.getElementById('paper-btn');
  const statusEl = document.getElementById('paper-status');
  const logEl = document.getElementById('paper-log');
  btn.disabled = true;
  statusEl.innerHTML = '<span class="status-dot dot-yellow"></span>Wird ausgeführt…';
  statusEl.style.color = '#facc15';
  logEl.textContent = '';
  _post('/api/paper/run')
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        statusEl.innerHTML = '<span class="status-dot dot-green"></span><span style="color:#4ade80">Abgeschlossen</span>';
      } else {
        statusEl.innerHTML = '<span class="status-dot dot-red"></span><span style="color:#f87171">Fehler</span>';
      }
      logEl.textContent = data.output || data.message || '';
      btn.disabled = false;
    })
    .catch(e => {
      statusEl.innerHTML = '<span class="status-dot dot-red"></span><span style="color:#f87171">Verbindungsfehler</span>';
      btn.disabled = false;
    });
}

// Sofort und dann alle 10 Sekunden Status prüfen
updateCollectorStatus();
updateTraderStatus();
updateTrainingStatus();
setInterval(updateCollectorStatus, 10000);
setInterval(updateTraderStatus, 10000);
setInterval(updateTrainingStatus, 15000);
</script>
</body>
</html>"""


# -------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------
def _get_signals() -> dict:
    try:
        df = get_recent_candles(SYMBOL, "5m", limit=200)
        if len(df) < 52:
            return {}
        df = calculate_all(df)
        return get_latest_signals(df)
    except Exception as e:
        logger.warning("Signale konnten nicht geladen werden: %s", e)
        return {}


_training_cache: dict = {"data": None, "ts": 0.0}
_TRAINING_CACHE_TTL = 300  # seconds


def _get_training_stats() -> dict:
    now = _time.monotonic()
    if _training_cache["data"] is not None and now - _training_cache["ts"] < _TRAINING_CACHE_TTL:
        return _training_cache["data"]

    result = {}
    for tf in ["1m", "5m", "1h"]:
        try:
            df = get_recent_candles(SYMBOL, tf, limit=5000)
            if len(df) < 10:
                result[tf] = {"candles": len(df), "clean_rows": 0, "sequences": 0,
                              "hold": 0, "buy": 0, "sell": 0, "ready": False}
                continue
            df = calculate_all(df)
            df = prepare_model_features(df)
            clean = df[FEATURE_COLUMNS + ["close"]].dropna().reset_index(drop=True)
            clean_len = len(clean)
            hold = buy = sell = 0
            if clean_len > LOOKAHEAD_BARS:
                closes = clean["close"].values
                for i in range(clean_len - LOOKAHEAD_BARS):
                    fr = (closes[i + LOOKAHEAD_BARS] - closes[i]) / closes[i]
                    if fr > LABEL_THRESHOLD_PCT:
                        buy += 1
                    elif fr < -LABEL_THRESHOLD_PCT:
                        sell += 1
                    else:
                        hold += 1
            sequences = max(0, clean_len - LOOKBACK_STEPS - LOOKAHEAD_BARS)
            result[tf] = {
                "candles": len(df),
                "clean_rows": clean_len,
                "sequences": sequences,
                "hold": hold,
                "buy": buy,
                "sell": sell,
                "ready": sequences >= MIN_TRAINING_SAMPLES,
            }
        except Exception as e:
            logger.warning("Trainingsstats für %s fehlgeschlagen: %s", tf, e)
            result[tf] = {"candles": 0, "clean_rows": 0, "sequences": 0,
                          "hold": 0, "buy": 0, "sell": 0, "ready": False}

    _training_cache["data"] = result
    _training_cache["ts"] = now
    return result


def _format_trades(raw: list) -> list:
    result = []
    for t in reversed(raw[-20:]):
        try:
            ts = datetime.fromtimestamp(t["timestamp"] / 1000).strftime("%d.%m %H:%M")
        except Exception:
            ts = "—"
        result.append({**t, "time": ts})
    return result


def _get_model_metrics() -> dict:
    try:
        with open(MODEL_METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _render_loss_chart(history: dict) -> str:
    losses = history.get("loss", [])
    val_losses = history.get("val_loss", [])
    if not losses:
        return ""

    W, H = 560, 170
    PL, PR, PT, PB = 48, 16, 16, 36
    cw = W - PL - PR
    ch = H - PT - PB
    n = len(losses)
    all_vals = losses + val_losses
    lo = min(all_vals) * 0.97
    hi = max(all_vals) * 1.03

    def sx(i):
        return PL + (i / max(n - 1, 1)) * cw

    def sy(v):
        return PT + ch - ((v - lo) / max(hi - lo, 1e-9)) * ch

    def polyline(vals, color):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'

    grid = ""
    for tick in [lo, (lo + hi) / 2, hi]:
        y = sy(tick)
        grid += (
            f'<line x1="{PL}" y1="{y:.1f}" x2="{W - PR}" y2="{y:.1f}" stroke="#1e2330" stroke-width="1"/>'
            f'<text x="{PL - 5}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#475569">{tick:.3f}</text>'
        )

    x_labels = ""
    step = max(1, n // 6)
    for i in range(0, n, step):
        x = sx(i)
        x_labels += f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle" font-size="10" fill="#475569">{i + 1}</text>'

    axes = (
        f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{H - PB}" stroke="#2d3748" stroke-width="1"/>'
        f'<line x1="{PL}" y1="{H - PB}" x2="{W - PR}" y2="{H - PB}" stroke="#2d3748" stroke-width="1"/>'
    )
    legend = (
        f'<rect x="{PL}" y="2" width="14" height="4" rx="2" fill="#38bdf8"/>'
        f'<text x="{PL + 18}" y="9" font-size="10" fill="#94a3b8">Train Loss</text>'
        f'<rect x="{PL + 88}" y="2" width="14" height="4" rx="2" fill="#f97316"/>'
        f'<text x="{PL + 106}" y="9" font-size="10" fill="#94a3b8">Val Loss</text>'
    )

    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-height:170px">'
        f'{grid}{axes}{x_labels}'
        f'{polyline(losses, "#38bdf8")}'
        f'{polyline(val_losses, "#f97316")}'
        f'{legend}'
        f'</svg>'
    )


def _collector_running() -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", "iota-collector"],
        capture_output=True,
    )
    return result.returncode == 0


# -------------------------------------------------------
# Auth-Routen
# -------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == DASHBOARD_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("index"))
        return render_template_string(_LOGIN_TEMPLATE, error="Falsches Passwort")
    return render_template_string(_LOGIN_TEMPLATE, error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -------------------------------------------------------
# Haupt-Route
# -------------------------------------------------------
@app.route("/")
@login_required
def index():
    try:
        status = get_portfolio_status()
    except Exception as e:
        logger.error("Portfolio konnte nicht geladen werden: %s", e)
        status = {
            "portfolio_value": 0, "cash": STARTING_CAPITAL, "iota_holdings": 0,
            "realized_pnl": 0, "unrealized_pnl": 0, "total_pnl": 0,
            "current_price": 0, "entry_price": None, "trade_count": 0,
        }

    trades = _format_trades(get_all_trades())
    signals = _get_signals()
    position = get_open_position()
    model_ready = os.path.exists(MODEL_PATH)
    stats = get_data_stats(SYMBOL)
    training_stats = _get_training_stats()
    metrics = _get_model_metrics()
    loss_chart = _render_loss_chart(metrics.get("history", {}))

    return render_template_string(
        _TEMPLATE,
        status=status,
        trades=trades,
        signals=signals,
        position=position,
        model_ready=model_ready,
        stats=stats,
        training_stats=training_stats,
        metrics=metrics,
        loss_chart=loss_chart,
        symbol=SYMBOL,
        start_capital=STARTING_CAPITAL,
        refresh=DASHBOARD_REFRESH_SECONDS,
        now=datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        min_samples=MIN_TRAINING_SAMPLES,
        lookback_steps=LOOKBACK_STEPS,
        lookahead_bars=LOOKAHEAD_BARS,
        label_threshold_pct=LABEL_THRESHOLD_PCT,
    )


# -------------------------------------------------------
# API — Collector
# -------------------------------------------------------
@app.route("/api/collector/status")
@login_required
def api_collector_status():
    return jsonify({"running": _collector_running()})


@app.route("/api/collector/start", methods=["POST"])
@login_required
def api_collector_start():
    subprocess.Popen(
        ["bash", str(SCRIPT_DIR / "start_collector.sh")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return jsonify({"ok": True})


@app.route("/api/collector/stop", methods=["POST"])
@login_required
def api_collector_stop():
    try:
        subprocess.Popen(
            ["bash", str(SCRIPT_DIR / "stop_collector.sh")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.error("stop_collector failed: %s", e)
        return jsonify({"ok": False, "message": str(e)})
    return jsonify({"ok": True})


# -------------------------------------------------------
# API — Trading Loop
# -------------------------------------------------------
def _trader_running() -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", "iota-trader"],
        capture_output=True,
    )
    return result.returncode == 0


@app.route("/api/trader/status")
@login_required
def api_trader_status():
    return jsonify({"running": _trader_running()})


@app.route("/api/trader/start", methods=["POST"])
@login_required
def api_trader_start():
    try:
        subprocess.Popen(
            ["bash", str(SCRIPT_DIR / "start_trader.sh")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.error("start_trader failed: %s", e)
        return jsonify({"ok": False, "message": str(e)})
    return jsonify({"ok": True})


@app.route("/api/trader/stop", methods=["POST"])
@login_required
def api_trader_stop():
    try:
        subprocess.Popen(
            ["bash", str(SCRIPT_DIR / "stop_trader.sh")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.error("stop_trader failed: %s", e)
        return jsonify({"ok": False, "message": str(e)})
    return jsonify({"ok": True})


# -------------------------------------------------------
# API — Training
# -------------------------------------------------------
def _training_running() -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", "iota-training"],
        capture_output=True,
    )
    return result.returncode == 0


@app.route("/api/training/status")
@login_required
def api_training_status():
    return jsonify({
        "running": _training_running(),
        "model_ready": os.path.exists(MODEL_PATH),
    })


@app.route("/api/training/start", methods=["POST"])
@login_required
def api_training_start():
    if _training_running():
        return jsonify({"ok": False, "message": "Training läuft bereits."})
    tf = request.args.get("timeframe", "1h")
    if tf not in ("1m", "5m", "1h"):
        return jsonify({"ok": False, "message": "Ungültiger Zeitrahmen."})
    try:
        subprocess.Popen(
            ["bash", str(SCRIPT_DIR / "start_training.sh"), tf],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.error("start_training failed: %s", e)
        return jsonify({"ok": False, "message": str(e)})
    return jsonify({"ok": True, "message": f"Training gestartet ({tf}). Kann einige Minuten dauern…"})


# -------------------------------------------------------
# API — Paper Trader
# -------------------------------------------------------
@app.route("/api/paper/run", methods=["POST"])
@login_required
def api_paper_run():
    try:
        result = subprocess.run(
            [PYTHON, str(SCRIPT_DIR / "main.py"), "paper"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(SCRIPT_DIR),
        )
        output = (result.stdout + result.stderr).strip()
        ok = result.returncode == 0
        return jsonify({"ok": ok, "output": output[-1500:] if len(output) > 1500 else output})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "output": "Timeout nach 60 Sekunden."})
    except Exception as e:
        return jsonify({"ok": False, "output": str(e)})


# -------------------------------------------------------
# Template filter
# -------------------------------------------------------
@app.template_filter("ts")
def _ts_filter(ms):
    if not ms:
        return "—"
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%d.%m.%y %H:%M")
    except Exception:
        return "—"


# -------------------------------------------------------
# Einstiegspunkt
# -------------------------------------------------------
def run():
    from database import setup_database
    setup_database()
    logger.info("Dashboard startet auf http://%s:%d", DASHBOARD_HOST, DASHBOARD_PORT)
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run()
