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
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
)
from telegram_notifier import notify_system, send_message as _tg_send

_TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
from database import get_all_trades, get_open_position, get_recent_candles, get_data_stats, reset_trades
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
  <title>Admin Login</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 min-h-screen flex items-center justify-center px-4">
  <div class="w-full max-w-sm">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl">
      <div class="text-center mb-8">
        <div class="inline-flex items-center justify-center w-12 h-12 rounded-full bg-sky-900/40 border border-sky-800 mb-4">
          <svg class="w-6 h-6 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
          </svg>
        </div>
        <h1 class="text-xl font-bold text-white">Admin Login</h1>
        <p class="text-sm text-slate-500 mt-1">Bitte anmelden um fortzufahren</p>
      </div>
      {% if error %}
      <div class="mb-4 px-4 py-2.5 rounded-lg bg-red-950/50 border border-red-900 text-red-400 text-sm text-center">
        {{ error }}
      </div>
      {% endif %}
      <form method="post" class="space-y-4">
        <input type="password" name="password" placeholder="Passwort" autofocus
               class="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-3 text-slate-200
                      placeholder-slate-600 focus:outline-none focus:border-sky-500 focus:ring-1
                      focus:ring-sky-500 transition-colors text-sm">
        <button type="submit"
                class="w-full bg-sky-600 hover:bg-sky-500 text-white font-semibold py-3 rounded-xl
                       transition-colors text-sm">
          Anmelden
        </button>
      </form>
    </div>
    <p class="text-center text-slate-700 text-xs mt-6">You are not authorized to access this page.</p>
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
  <title>IOTA Trading Bot</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      safelist: [
        'text-emerald-400','text-amber-400','text-red-400','text-sky-400','text-slate-400',
        'bg-emerald-400','bg-amber-400','bg-red-400','bg-sky-400','bg-slate-500','bg-slate-600',
        'border-sky-400','animate-pulse',
        'bg-emerald-900/40','bg-red-900/40','bg-violet-900/40',
        'border-emerald-900','border-red-900','border-violet-900',
        'text-violet-400',
      ]
    }
  </script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js"></script>
  <script>
  function dashboard() {
    return {
      activeTab: 'kurs',
      collector:   { running: null },
      trader:      { running: null },
      training:    { running: null, model_ready: false },
      history:     { running: null },
      paper:       { loading: false },
      paperLog:    '',
      historyLog:  '',
      trainingLog: '',
      sim: { loading: false, tf: '1h', limit: 500, threshold: 0.45, data: null, error: '', chart: null },
      priceChart:   null,
      priceChartTf: '1h',
      priceLoading: false,

      init() {
        this.pollAll();
        setInterval(() => this.pollAll(), 10000);
        setTimeout(() => this.initChart(), 200);
        setTimeout(() => this.loadPriceChart('1h'), 300);
        this.$watch('activeTab', val => {
          if (val === 'kurs') this.$nextTick(() => this.loadPriceChart());
        });
      },

      async pollAll() {
        await Promise.all([
          this.pollCollector(),
          this.pollTrader(),
          this.pollTraining(),
          this.pollHistory(),
        ]);
      },

      async pollCollector() {
        try {
          const d = await fetch('/api/collector/status').then(r => r.json());
          this.collector.running = d.running;
        } catch { this.collector.running = null; }
      },

      async pollTrader() {
        try {
          const d = await fetch('/api/trader/status').then(r => r.json());
          this.trader.running = d.running;
        } catch { this.trader.running = null; }
      },

      async pollTraining() {
        try {
          const d = await fetch('/api/training/status').then(r => r.json());
          this.training.running = d.running;
          this.training.model_ready = d.model_ready;
        } catch { this.training.running = null; }
      },

      async pollHistory() {
        try {
          const d = await fetch('/api/history/status').then(r => r.json());
          this.history.running = d.running;
        } catch { this.history.running = null; }
      },

      async collectorAction(action) {
        await fetch('/api/collector/' + action, { method: 'POST' });
        setTimeout(() => this.pollCollector(), 1800);
      },

      async traderAction(action) {
        await fetch('/api/trader/' + action, { method: 'POST' });
        setTimeout(() => this.pollTrader(), 1800);
      },

      async startHistory() {
        const tf   = document.getElementById('hist-tf').value;
        const days = document.getElementById('hist-days').value;
        this.historyLog = 'Starte Download (' + tf + ', ' + days + ' Tage)…';
        const d = await fetch('/api/history/start?timeframe=' + tf + '&days=' + days, { method: 'POST' }).then(r => r.json());
        this.historyLog = d.message || '';
        await this.pollHistory();
        const poll = setInterval(async () => {
          await this.pollHistory();
          if (!this.history.running) {
            clearInterval(poll);
            this.historyLog = '\\u2713 Download abgeschlossen.';
          }
        }, 4000);
      },

      async startTraining() {
        const tf = document.getElementById('train-tf').value;
        this.trainingLog = 'Starte Training (' + tf + ')…';
        const d = await fetch('/api/training/start?timeframe=' + tf, { method: 'POST' }).then(r => r.json());
        this.trainingLog = d.message || '';
        await this.pollTraining();
        const poll = setInterval(async () => {
          await this.pollTraining();
          if (!this.training.running) {
            clearInterval(poll);
            this.trainingLog = this.training.model_ready
              ? '\\u2713 Training abgeschlossen — Modell gespeichert.'
              : '\\u2717 Training beendet (kein Modell gefunden).';
          }
        }, 5000);
      },

      async runPaper() {
        this.paper.loading = true;
        this.paperLog = '';
        try {
          const d = await fetch('/api/paper/run', { method: 'POST' }).then(r => r.json());
          this.paperLog = d.output || d.message || '';
        } catch(e) {
          this.paperLog = 'Verbindungsfehler: ' + e.message;
        }
        this.paper.loading = false;
      },

      initChart() {
        const canvas = document.getElementById('loss-chart');
        if (!canvas) return;
        const history = {{ history_json | safe }};
        const losses    = history.loss     || [];
        const valLosses = history.val_loss || [];
        if (!losses.length) return;
        new Chart(canvas, {
          type: 'line',
          data: {
            labels: losses.map((_, i) => i + 1),
            datasets: [
              {
                label: 'Train Loss',
                data: losses,
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56,189,248,0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.35,
              },
              {
                label: 'Val Loss',
                data: valLosses,
                borderColor: '#fb923c',
                backgroundColor: 'rgba(251,146,60,0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.35,
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
              legend: { labels: { color: '#94a3b8', boxWidth: 10, font: { size: 11 } } }
            },
            scales: {
              x: {
                ticks: { color: '#475569', maxTicksLimit: 10, font: { size: 10 } },
                grid: { color: '#1e293b' },
                title: { display: true, text: 'Epoche', color: '#475569', font: { size: 10 } }
              },
              y: {
                ticks: { color: '#475569', font: { size: 10 } },
                grid: { color: '#1e293b' },
                title: { display: true, text: 'Loss', color: '#475569', font: { size: 10 } }
              }
            }
          }
        });
      },

      async loadPriceChart(tf) {
        tf = tf || this.priceChartTf;
        this.priceChartTf = tf;
        this.priceLoading = true;
        try {
          const d = await fetch('/api/chart/data?timeframe=' + tf + '&limit=300').then(r => r.json());
          const canvas = document.getElementById('price-chart');
          if (!canvas || !d.close.length) { this.priceLoading = false; return; }
          if (this.priceChart) {
            this.priceChart.data.labels = d.labels;
            this.priceChart.data.datasets[0].data = d.close;
            this.priceChart.data.datasets[1].data = d.high;
            this.priceChart.data.datasets[2].data = d.low;
            this.priceChart.update('none');
          } else {
            const ctx = canvas.getContext('2d');
            const grad = ctx.createLinearGradient(0, 0, 0, 340);
            grad.addColorStop(0, 'rgba(56,189,248,0.18)');
            grad.addColorStop(1, 'rgba(56,189,248,0)');
            this.priceChart = new Chart(canvas, {
              type: 'line',
              data: {
                labels: d.labels,
                datasets: [
                  {
                    label: 'Close',
                    data: d.close,
                    borderColor: '#38bdf8',
                    backgroundColor: grad,
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                    order: 1,
                  },
                  {
                    label: 'High',
                    data: d.high,
                    borderColor: 'rgba(52,211,153,0.35)',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                    order: 2,
                  },
                  {
                    label: 'Low',
                    data: d.low,
                    borderColor: 'rgba(248,113,113,0.35)',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                    order: 3,
                  }
                ]
              },
              options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                  legend: { labels: { color: '#475569', boxWidth: 10, font: { size: 11 } } },
                  tooltip: {
                    callbacks: {
                      label: ctx => ctx.dataset.label + ': $' + ctx.raw.toFixed(4)
                    }
                  }
                },
                scales: {
                  x: {
                    ticks: { color: '#475569', maxTicksLimit: 8, font: { size: 10 } },
                    grid: { color: '#1e293b' },
                  },
                  y: {
                    ticks: { color: '#475569', font: { size: 10 }, callback: v => '$' + v.toFixed(3) },
                    grid: { color: '#1e293b' },
                    position: 'right',
                  }
                }
              }
            });
          }
        } catch(e) { console.error('Chart load failed', e); }
        this.priceLoading = false;
      },

      async runSimulation() {
        this.sim.loading = true;
        this.sim.error   = '';
        this.sim.data    = null;
        if (this.sim.chart) { this.sim.chart.destroy(); this.sim.chart = null; }
        try {
          const r = await fetch(
            '/api/simulation/run?timeframe=' + this.sim.tf + '&limit=' + this.sim.limit + '&threshold=' + this.sim.threshold,
            { method: 'POST' }
          );
          const d = await r.json();
          if (d.error) { this.sim.error = d.error; }
          else { this.sim.data = d; this.$nextTick(() => this.renderSimChart(d)); }
        } catch(e) { this.sim.error = 'Verbindungsfehler: ' + e.message; }
        this.sim.loading = false;
      },

      renderSimChart(data) {
        const canvas = document.getElementById('sim-chart');
        if (!canvas) return;
        if (this.sim.chart) { this.sim.chart.destroy(); this.sim.chart = null; }
        const bpRadii = data.buy_prices.map(v  => v !== null ? 7 : 0);
        const spRadii = data.sell_prices.map(v => v !== null ? 7 : 0);
        this.sim.chart = new Chart(canvas, {
          type: 'line',
          data: {
            labels: data.labels,
            datasets: [
              {
                label: 'Portfolio $',
                data: data.equity,
                borderColor: '#34d399',
                backgroundColor: 'rgba(52,211,153,0.06)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.3,
                yAxisID: 'yEq',
                order: 3,
              },
              {
                label: 'Preis',
                data: data.closes,
                borderColor: '#38bdf8',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0.3,
                yAxisID: 'yPx',
                order: 4,
              },
              {
                label: 'BUY',
                data: data.buy_prices,
                backgroundColor: '#4ade80',
                borderColor: '#4ade80',
                pointRadius: bpRadii,
                pointHoverRadius: bpRadii.map(r => r ? r + 2 : 0),
                pointStyle: 'triangle',
                showLine: false,
                yAxisID: 'yPx',
                order: 1,
              },
              {
                label: 'SELL',
                data: data.sell_prices,
                backgroundColor: '#f87171',
                borderColor: '#f87171',
                pointRadius: spRadii,
                pointHoverRadius: spRadii.map(r => r ? r + 2 : 0),
                pointStyle: 'triangle',
                rotation: 180,
                showLine: false,
                yAxisID: 'yPx',
                order: 2,
              },
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
              legend: { labels: { color: '#94a3b8', boxWidth: 10, font: { size: 11 },
                filter: item => item.text !== 'Preis' || true } },
              tooltip: {
                callbacks: {
                  label: ctx => {
                    if (ctx.raw === null || ctx.raw === undefined) return null;
                    if (ctx.dataset.label === 'Portfolio $') return 'Portfolio: $' + ctx.raw.toFixed(2);
                    if (ctx.dataset.label === 'Preis') return 'Preis: $' + ctx.raw.toFixed(4);
                    if (ctx.dataset.label === 'BUY')  return '\\u25b2 BUY @ $' + ctx.raw.toFixed(4);
                    if (ctx.dataset.label === 'SELL') return '\\u25bc SELL @ $' + ctx.raw.toFixed(4);
                    return null;
                  }
                }
              }
            },
            scales: {
              x: {
                ticks: { color: '#475569', maxTicksLimit: 10, font: { size: 10 } },
                grid:  { color: '#1e293b' },
              },
              yPx: {
                type: 'linear', position: 'right',
                ticks: { color: '#38bdf8', font: { size: 10 }, callback: v => '$' + v.toFixed(4) },
                grid:  { color: '#1e293b' },
                title: { display: true, text: 'Preis', color: '#38bdf8', font: { size: 10 } },
              },
              yEq: {
                type: 'linear', position: 'left',
                ticks: { color: '#34d399', font: { size: 10 }, callback: v => '$' + v.toFixed(0) },
                grid:  { display: false },
                title: { display: true, text: 'Portfolio', color: '#34d399', font: { size: 10 } },
              },
            }
          }
        });
      },

      async testTelegram() {
        const d = await fetch('/api/telegram/test', { method: 'POST' }).then(r => r.json());
        alert(d.message || (d.ok ? 'Gesendet!' : 'Fehler'));
      },

      async resetPortfolio() {
        if (!confirm('Alle Trades löschen und Portfolio auf Startkapital zurücksetzen?\\n\\nDiese Aktion kann nicht rückgängig gemacht werden.')) return;
        try {
          const d = await fetch('/api/paper/reset', { method: 'POST' }).then(r => r.json());
          if (d.ok) {
            alert(d.deleted + ' Trade(s) gelöscht. Portfolio wurde zurückgesetzt.');
            location.reload();
          } else {
            alert('Fehler: ' + (d.message || 'Unbekannt'));
          }
        } catch(e) {
          alert('Verbindungsfehler: ' + e.message);
        }
      },

      dotClass(v) {
        if (v === null) return 'bg-amber-400 animate-pulse';
        return v ? 'bg-emerald-400' : 'bg-slate-600';
      },
      statusText(v) {
        if (v === null) return 'Prüfe…';
        return v ? 'Aktiv' : 'Gestoppt';
      },
      statusColor(v) {
        if (v === null) return 'text-amber-400';
        return v ? 'text-emerald-400' : 'text-slate-400';
      }
    }
  }
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js"></script>
  <style>
    [x-cloak] { display: none !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-200 min-h-screen" x-data="dashboard()" x-init="init()">

<!-- ===== HEADER ===== -->
<header class="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/70">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-4">
    <div class="flex-shrink-0">
      <span class="font-bold text-white text-sm tracking-tight">IOTA Bot</span>
      <span class="hidden sm:inline text-slate-600 text-xs ml-2">{{ symbol }}</span>
    </div>

    <!-- status pills -->
    <div class="flex items-center gap-2 flex-1 overflow-x-auto no-scrollbar">
      <span class="inline-flex items-center gap-1.5 bg-slate-900 border border-slate-800 rounded-full px-2.5 py-1 text-xs whitespace-nowrap">
        <span class="w-1.5 h-1.5 rounded-full flex-shrink-0" :class="dotClass(collector.running)"></span>
        <span class="text-slate-400">Collector</span>
        <span class="font-medium" :class="statusColor(collector.running)" x-text="statusText(collector.running)"></span>
      </span>
      <span class="inline-flex items-center gap-1.5 bg-slate-900 border border-slate-800 rounded-full px-2.5 py-1 text-xs whitespace-nowrap">
        <span class="w-1.5 h-1.5 rounded-full flex-shrink-0" :class="dotClass(trader.running)"></span>
        <span class="text-slate-400">Trader</span>
        <span class="font-medium" :class="statusColor(trader.running)" x-text="statusText(trader.running)"></span>
      </span>
      <span class="inline-flex items-center gap-1.5 bg-slate-900 border border-slate-800 rounded-full px-2.5 py-1 text-xs whitespace-nowrap">
        <span class="w-1.5 h-1.5 rounded-full flex-shrink-0"
              :class="training.running ? 'bg-amber-400 animate-pulse' : (training.model_ready ? 'bg-emerald-400' : 'bg-slate-600')"></span>
        <span class="text-slate-400">Modell</span>
        <span class="font-medium"
              :class="training.running ? 'text-amber-400' : (training.model_ready ? 'text-emerald-400' : 'text-slate-400')"
              x-text="training.running ? 'Training…' : (training.model_ready ? 'Bereit' : 'Fehlt')"></span>
      </span>
      {% if telegram_enabled %}
      <button @click="testTelegram()"
              class="inline-flex items-center gap-1.5 bg-slate-900 border border-sky-900/50
                     hover:border-sky-700 rounded-full px-2.5 py-1 text-xs whitespace-nowrap
                     transition-colors cursor-pointer">
        <span class="w-1.5 h-1.5 rounded-full bg-sky-400"></span>
        <span class="text-slate-400">Telegram</span>
        <span class="font-medium text-sky-400">Aktiv</span>
      </button>
      {% else %}
      <span class="inline-flex items-center gap-1.5 bg-slate-900 border border-slate-800
                   rounded-full px-2.5 py-1 text-xs whitespace-nowrap">
        <span class="w-1.5 h-1.5 rounded-full bg-slate-600"></span>
        <span class="text-slate-600">Telegram</span>
        <span class="font-medium text-slate-600">Inaktiv</span>
      </span>
      {% endif %}
      <span class="text-slate-700 text-xs hidden sm:inline ml-auto">{{ now }}</span>
    </div>

    <a href="/logout"
       class="flex-shrink-0 text-xs text-slate-500 hover:text-red-400 border border-slate-800 hover:border-red-900/60
              rounded-lg px-3 py-1.5 transition-colors whitespace-nowrap">
      Abmelden
    </a>
  </div>
</header>

<main class="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-8">

  <!-- ===== PORTFOLIO ===== -->
  <section>
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-xs font-semibold uppercase tracking-widest text-slate-500">Portfolio</h2>
      <button @click="resetPortfolio()"
              class="text-xs text-red-500 hover:text-red-400 border border-red-900/40 hover:border-red-700
                     rounded-lg px-3 py-1.5 transition-colors flex items-center gap-1.5">
        &#8635; Reset
      </button>
    </div>
    <div class="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-8 gap-3">

      <div class="col-span-2 bg-gradient-to-br from-sky-900/30 to-slate-900 border border-sky-800/40 rounded-2xl p-5">
        <div class="text-xs text-slate-500 uppercase tracking-wider mb-1">Portfolio-Wert</div>
        <div class="text-3xl font-bold text-sky-400">${{ status.portfolio_value }}</div>
        <div class="text-xs text-slate-600 mt-1.5">Startkapital: ${{ "%.2f"|format(start_capital) }}</div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Cash</div>
        <div class="text-xl font-bold text-slate-100">${{ status.cash }}</div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">IOTA-Bestand</div>
        <div class="text-xl font-bold text-slate-100">{{ status.iota_holdings }}</div>
        <div class="text-xs text-slate-600 mt-1">IOTA</div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Real. PnL</div>
        <div class="text-xl font-bold {{ 'text-emerald-400' if status.realized_pnl >= 0 else 'text-red-400' }}">
          {{ '+' if status.realized_pnl >= 0 else '' }}${{ status.realized_pnl }}
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Unreal. PnL</div>
        <div class="text-xl font-bold {{ 'text-emerald-400' if status.unrealized_pnl >= 0 else 'text-red-400' }}">
          {{ '+' if status.unrealized_pnl >= 0 else '' }}${{ status.unrealized_pnl }}
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Preis</div>
        <div class="text-xl font-bold text-amber-400">${{ status.current_price }}</div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
        <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Trades / Position</div>
        <div class="text-xl font-bold text-slate-100">{{ status.trade_count }}</div>
        <div class="text-xs mt-1.5 {{ 'text-emerald-400' if position else 'text-slate-600' }}">
          {% if position %}
            &#9679; OFFEN @ ${{ "%.4f"|format(position.price) }}
          {% else %}
            Keine Position
          {% endif %}
        </div>
      </div>

    </div>
  </section>

  <!-- ===== STEUERUNG ===== -->
  <section>
    <h2 class="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Steuerung</h2>
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">

      <!-- Collector -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Datensammler</div>
        <div class="flex items-center gap-2 mb-4 h-6">
          <span class="w-2 h-2 rounded-full flex-shrink-0 transition-colors" :class="dotClass(collector.running)"></span>
          <span class="text-sm font-medium transition-colors" :class="statusColor(collector.running)" x-text="statusText(collector.running)"></span>
        </div>
        <div class="flex gap-2">
          <button @click="collectorAction('start')"
                  :disabled="collector.running === true"
                  class="flex-1 py-2 text-xs font-semibold rounded-xl bg-emerald-900/40 text-emerald-400
                         border border-emerald-900 hover:bg-emerald-900/70 disabled:opacity-30
                         disabled:cursor-not-allowed transition-all">
            ▶ Start
          </button>
          <button @click="collectorAction('stop')"
                  :disabled="collector.running !== true"
                  class="flex-1 py-2 text-xs font-semibold rounded-xl bg-red-900/40 text-red-400
                         border border-red-900 hover:bg-red-900/70 disabled:opacity-30
                         disabled:cursor-not-allowed transition-all">
            ■ Stop
          </button>
        </div>
      </div>

      <!-- Trader -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Trading-Loop</div>
        <div class="flex items-center gap-2 mb-4 h-6">
          <span class="w-2 h-2 rounded-full flex-shrink-0 transition-colors" :class="dotClass(trader.running)"></span>
          <span class="text-sm font-medium transition-colors" :class="statusColor(trader.running)" x-text="statusText(trader.running)"></span>
        </div>
        <div class="flex gap-2">
          <button @click="traderAction('start')"
                  :disabled="trader.running === true"
                  class="flex-1 py-2 text-xs font-semibold rounded-xl bg-emerald-900/40 text-emerald-400
                         border border-emerald-900 hover:bg-emerald-900/70 disabled:opacity-30
                         disabled:cursor-not-allowed transition-all">
            ▶ Start
          </button>
          <button @click="traderAction('stop')"
                  :disabled="trader.running !== true"
                  class="flex-1 py-2 text-xs font-semibold rounded-xl bg-red-900/40 text-red-400
                         border border-red-900 hover:bg-red-900/70 disabled:opacity-30
                         disabled:cursor-not-allowed transition-all">
            ■ Stop
          </button>
        </div>
      </div>

      <!-- Paper Trader -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Paper Trader</div>
        <div class="text-xs text-slate-500 mb-4 h-6 flex items-center">Einmalige Iteration ausführen</div>
        <button @click="runPaper()"
                :disabled="paper.loading"
                class="w-full py-2 text-xs font-semibold rounded-xl bg-sky-900/40 text-sky-400
                       border border-sky-900 hover:bg-sky-900/70 disabled:opacity-30
                       disabled:cursor-not-allowed transition-all flex items-center justify-center gap-1.5">
          <span x-show="!paper.loading">&#9889; Ausführen</span>
          <span x-show="paper.loading" class="flex items-center gap-1.5">
            <svg class="animate-spin w-3 h-3" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            Läuft…
          </span>
        </button>
        <div x-show="paperLog" x-text="paperLog"
             class="mt-2 text-xs text-slate-500 font-mono bg-slate-950 rounded-lg p-2
                    whitespace-pre-wrap break-all max-h-24 overflow-y-auto leading-relaxed"></div>
      </div>

      <!-- Historical download -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Historische Daten</div>
        <div class="flex items-center gap-2 mb-3 h-6">
          <span class="w-2 h-2 rounded-full flex-shrink-0"
                :class="history.running ? 'bg-amber-400 animate-pulse' : 'bg-slate-600'"></span>
          <span class="text-sm"
                :class="history.running ? 'text-amber-400' : 'text-slate-400'"
                x-text="history.running ? 'Lädt…' : 'Bereit'"></span>
        </div>
        <div class="flex gap-2 mb-2">
          <select id="hist-tf"
                  class="flex-1 bg-slate-950 border border-slate-700 rounded-lg text-xs text-slate-300
                         px-2 py-1.5 focus:outline-none focus:border-sky-600 transition-colors">
            <option value="1h">1h</option>
            <option value="5m">5m</option>
            <option value="1m">1m</option>
          </select>
          <input id="hist-days" type="number" value="365" min="1" max="1825"
                 class="w-16 bg-slate-950 border border-slate-700 rounded-lg text-xs text-slate-300
                        px-2 py-1.5 focus:outline-none focus:border-sky-600 transition-colors text-center">
          <span class="text-xs text-slate-600 self-center">d</span>
        </div>
        <button @click="startHistory()"
                :disabled="history.running"
                class="w-full py-2 text-xs font-semibold rounded-xl bg-sky-900/40 text-sky-400
                       border border-sky-900 hover:bg-sky-900/70 disabled:opacity-30
                       disabled:cursor-not-allowed transition-all">
          &#128229; Laden
        </button>
        <div x-show="historyLog" x-text="historyLog"
             class="mt-2 text-xs text-slate-500 font-mono bg-slate-950 rounded-lg p-2"></div>
      </div>

      <!-- Training -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Modell-Training</div>
        <div class="flex items-center gap-2 mb-3 h-6">
          <span class="w-2 h-2 rounded-full flex-shrink-0"
                :class="training.running ? 'bg-amber-400 animate-pulse' : (training.model_ready ? 'bg-emerald-400' : 'bg-slate-600')"></span>
          <span class="text-sm font-medium"
                :class="training.running ? 'text-amber-400' : (training.model_ready ? 'text-emerald-400' : 'text-slate-400')"
                x-text="training.running ? 'Läuft…' : (training.model_ready ? 'Modell bereit' : 'Kein Modell')"></span>
        </div>
        <select id="train-tf"
                class="w-full bg-slate-950 border border-slate-700 rounded-lg text-xs text-slate-300
                       px-2 py-1.5 mb-2 focus:outline-none focus:border-sky-600 transition-colors">
          <option value="1h">1h</option>
          <option value="5m">5m</option>
          <option value="1m">1m</option>
        </select>
        <button @click="startTraining()"
                :disabled="training.running"
                class="w-full py-2 text-xs font-semibold rounded-xl bg-violet-900/40 text-violet-400
                       border border-violet-900 hover:bg-violet-900/70 disabled:opacity-30
                       disabled:cursor-not-allowed transition-all">
          &#129504; Training starten
        </button>
        <div x-show="trainingLog" x-text="trainingLog"
             class="mt-2 text-xs text-slate-500 font-mono bg-slate-950 rounded-lg p-2"></div>
      </div>

    </div>
  </section>

  <!-- ===== TABS ===== -->
  <section>
    <!-- Tab bar -->
    <div class="flex gap-0 border-b border-slate-800 mb-5 -mx-1 overflow-x-auto no-scrollbar">
      <button @click="activeTab = 'kurs'"
              :class="activeTab === 'kurs'
                ? 'text-sky-400 border-b-2 border-sky-400 bg-sky-400/5'
                : 'text-slate-500 hover:text-slate-300 border-b-2 border-transparent'"
              class="px-5 py-2.5 text-sm font-medium transition-colors -mb-px rounded-t-lg flex-shrink-0">
        Kurschart
      </button>
      <button @click="activeTab = 'signals'"
              :class="activeTab === 'signals'
                ? 'text-sky-400 border-b-2 border-sky-400 bg-sky-400/5'
                : 'text-slate-500 hover:text-slate-300 border-b-2 border-transparent'"
              class="px-5 py-2.5 text-sm font-medium transition-colors -mb-px rounded-t-lg flex-shrink-0">
        Signale
      </button>
      <button @click="activeTab = 'model'"
              :class="activeTab === 'model'
                ? 'text-sky-400 border-b-2 border-sky-400 bg-sky-400/5'
                : 'text-slate-500 hover:text-slate-300 border-b-2 border-transparent'"
              class="px-5 py-2.5 text-sm font-medium transition-colors -mb-px rounded-t-lg flex-shrink-0">
        Modell
      </button>
      <button @click="activeTab = 'data'"
              :class="activeTab === 'data'
                ? 'text-sky-400 border-b-2 border-sky-400 bg-sky-400/5'
                : 'text-slate-500 hover:text-slate-300 border-b-2 border-transparent'"
              class="px-5 py-2.5 text-sm font-medium transition-colors -mb-px rounded-t-lg flex-shrink-0">
        Daten
      </button>
      <button @click="activeTab = 'trades'"
              :class="activeTab === 'trades'
                ? 'text-sky-400 border-b-2 border-sky-400 bg-sky-400/5'
                : 'text-slate-500 hover:text-slate-300 border-b-2 border-transparent'"
              class="px-5 py-2.5 text-sm font-medium transition-colors -mb-px rounded-t-lg flex-shrink-0">
        Trades
        {% if trades %}<span class="ml-1.5 bg-slate-800 text-slate-400 text-xs rounded-full px-1.5 py-0.5">{{ trades|length }}</span>{% endif %}
      </button>
      <button @click="activeTab = 'sim'"
              :class="activeTab === 'sim'
                ? 'text-sky-400 border-b-2 border-sky-400 bg-sky-400/5'
                : 'text-slate-500 hover:text-slate-300 border-b-2 border-transparent'"
              class="px-5 py-2.5 text-sm font-medium transition-colors -mb-px rounded-t-lg flex-shrink-0">
        Simulation
      </button>
    </div>

    <!-- ── Tab: Kurs ── -->
    <div x-show="activeTab === 'kurs'" x-transition:enter="transition ease-out duration-150"
         x-transition:enter-start="opacity-0 translate-y-1" x-transition:enter-end="opacity-100 translate-y-0">
      <!-- Timeframe selector -->
      <div class="flex items-center gap-2 mb-4">
        <span class="text-xs text-slate-500">Zeitrahmen:</span>
        <template x-for="tf in ['1m','5m','1h']" :key="tf">
          <button @click="loadPriceChart(tf)"
                  :class="priceChartTf === tf
                    ? 'bg-sky-900/50 text-sky-400 border-sky-700'
                    : 'bg-slate-900 text-slate-500 border-slate-800 hover:text-slate-300'"
                  class="px-3 py-1 text-xs font-semibold rounded-lg border transition-colors"
                  x-text="tf"></button>
        </template>
        <span x-show="priceLoading" class="text-xs text-slate-500 flex items-center gap-1.5 ml-1">
          <svg class="animate-spin w-3 h-3" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
          Lädt…
        </span>
        <span class="ml-auto text-xs text-slate-600">IOTA/USD &bull; Bitfinex &bull; letzte 300 Bars</span>
      </div>
      <!-- Chart canvas -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div style="height: 360px; position: relative;">
          <canvas id="price-chart"></canvas>
          <div x-show="!priceLoading && priceChart === null"
               class="absolute inset-0 flex items-center justify-center">
            <div class="text-center">
              <div class="text-3xl mb-3">📡</div>
              <p class="text-slate-500 text-sm">Noch keine Kursdaten — bitte zuerst den Datensammler starten.</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ── Tab: Signale ── -->
    <div x-show="activeTab === 'signals'" x-transition:enter="transition ease-out duration-150"
         x-transition:enter-start="opacity-0 translate-y-1" x-transition:enter-end="opacity-100 translate-y-0">
      {% if signals %}
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">Preis</div>
          <div class="text-xl font-bold text-amber-400">${{ "%.4f"|format(signals.last_price) if signals.last_price else '—' }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">RSI 14</div>
          <div class="text-xl font-bold {{ 'text-red-400' if signals.rsi and signals.rsi > 70 else ('text-emerald-400' if signals.rsi and signals.rsi < 30 else 'text-slate-100') }}">
            {{ "%.1f"|format(signals.rsi) if signals.rsi else '—' }}
          </div>
          {% if signals.rsi %}
          <div class="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div class="h-full rounded-full transition-all {{ 'bg-red-400' if signals.rsi > 70 else ('bg-emerald-400' if signals.rsi < 30 else 'bg-sky-400') }}"
                 style="width: {{ [signals.rsi, 100]|min }}%"></div>
          </div>
          {% endif %}
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">SMA 20</div>
          <div class="text-xl font-bold text-slate-100">${{ "%.4f"|format(signals.sma_20) if signals.sma_20 else '—' }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">SMA 50</div>
          <div class="text-xl font-bold text-slate-100">${{ "%.4f"|format(signals.sma_50) if signals.sma_50 else '—' }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">EMA 20</div>
          <div class="text-xl font-bold text-slate-100">${{ "%.4f"|format(signals.ema_20) if signals.ema_20 else '—' }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">MACD</div>
          <div class="text-xl font-bold {{ 'text-emerald-400' if signals.macd and signals.macd > 0 else 'text-red-400' }}">
            {{ "%.5f"|format(signals.macd) if signals.macd else '—' }}
          </div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">MACD Signal</div>
          <div class="text-xl font-bold text-slate-100">{{ "%.5f"|format(signals.macd_signal) if signals.macd_signal else '—' }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">Volatilität</div>
          <div class="text-xl font-bold text-slate-100">{{ "%.5f"|format(signals.volatility) if signals.volatility else '—' }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-2">Änderung %</div>
          <div class="text-xl font-bold {{ 'text-emerald-400' if signals.pct_change and signals.pct_change > 0 else 'text-red-400' }}">
            {{ '%+.3f%%'|format(signals.pct_change * 100) if signals.pct_change else '—' }}
          </div>
        </div>
      </div>
      <p class="text-xs text-slate-600 mt-3">Zeitrahmen: 5m &bull; Aktualisierung alle {{ refresh }}s</p>
      {% else %}
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
        <div class="text-3xl mb-3">📡</div>
        <p class="text-slate-500 text-sm">Noch keine Signaldaten — bitte zuerst den Datensammler starten.</p>
      </div>
      {% endif %}
    </div>

    <!-- ── Tab: Modell ── -->
    <div x-show="activeTab === 'model'" x-transition:enter="transition ease-out duration-150"
         x-transition:enter-start="opacity-0 translate-y-1" x-transition:enter-end="opacity-100 translate-y-0">
      {% if metrics %}
      <!-- Metric cards -->
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Val Loss</div>
          <div class="text-2xl font-bold {{ 'text-emerald-400' if metrics.val_loss < 1.0 else ('text-amber-400' if metrics.val_loss < 1.2 else 'text-red-400') }}">
            {{ "%.4f"|format(metrics.val_loss) }}
          </div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Val Accuracy</div>
          <div class="text-2xl font-bold {{ 'text-emerald-400' if metrics.val_accuracy >= 0.45 else ('text-amber-400' if metrics.val_accuracy >= 0.35 else 'text-red-400') }}">
            {{ "%.1f"|format(metrics.val_accuracy * 100) }}%
          </div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Epochen</div>
          <div class="text-2xl font-bold text-slate-300">{{ metrics.epochs_ran }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Zeitrahmen</div>
          <div class="text-2xl font-bold text-sky-400">{{ metrics.timeframe }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Train-Samples</div>
          <div class="text-2xl font-bold text-slate-300">{{ metrics.train_samples }}</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Val-Samples</div>
          <div class="text-2xl font-bold text-slate-300">{{ metrics.val_samples }}</div>
        </div>
      </div>
      {% if metrics.class_weights %}
      <div class="bg-slate-900 border border-slate-800 rounded-2xl px-5 py-3 mb-5 flex flex-wrap items-center gap-x-6 gap-y-2">
        <span class="text-xs text-slate-500">Klassen-Gewichte</span>
        <span class="text-xs"><span class="text-slate-500">HOLD</span> <span class="font-mono font-semibold text-slate-300">{{ metrics.class_weights.HOLD }}</span></span>
        <span class="text-xs"><span class="text-slate-500">BUY</span> <span class="font-mono font-semibold text-emerald-400">{{ metrics.class_weights.BUY }}</span></span>
        <span class="text-xs"><span class="text-slate-500">SELL</span> <span class="font-mono font-semibold text-red-400">{{ metrics.class_weights.SELL }}</span></span>
        <span class="text-xs text-slate-600 ml-auto">Trainiert: <span class="text-sky-500">{{ metrics.trained_at }}</span></span>
      </div>
      {% endif %}
      <!-- Loss chart -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div class="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-4">Loss-Kurve</div>
        <div style="height: 220px; position: relative;">
          <canvas id="loss-chart"></canvas>
        </div>
      </div>
      {% else %}
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
        <div class="text-3xl mb-3">🧠</div>
        <p class="text-slate-500 text-sm">Noch kein Modell trainiert — Training über den Button in der Steuerung starten.</p>
      </div>
      {% endif %}

      <!-- Training stats table -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden mt-5">
        <div class="px-5 py-4 border-b border-slate-800">
          <span class="text-xs font-semibold uppercase tracking-wider text-slate-500">Trainingsdaten</span>
          <span class="text-xs text-slate-600 ml-4">
            Lookback: <span class="text-slate-400">{{ lookback_steps }}</span> &bull;
            Lookahead: <span class="text-slate-400">{{ lookahead_bars }}</span> &bull;
            Schwelle: <span class="text-slate-400">&#177;{{ "%.1f"|format(label_threshold_pct * 100) }}%</span>
          </span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-xs uppercase tracking-wider text-slate-500 border-b border-slate-800">
                <th class="text-left px-5 py-3">TF</th>
                <th class="text-right px-4 py-3">Candles</th>
                <th class="text-right px-4 py-3">Clean</th>
                <th class="text-right px-4 py-3">Seqs</th>
                <th class="text-right px-4 py-3 text-slate-400">HOLD</th>
                <th class="text-right px-4 py-3 text-emerald-600">BUY</th>
                <th class="text-right px-4 py-3 text-red-600">SELL</th>
                <th class="text-left px-5 py-3">Status</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800/50">
              {% for tf, d in training_stats.items() %}
              <tr class="hover:bg-slate-800/30 transition-colors">
                <td class="px-5 py-3.5 font-bold text-sky-400">{{ tf }}</td>
                <td class="px-4 py-3.5 text-right text-slate-300 font-mono text-xs">{{ d.candles }}</td>
                <td class="px-4 py-3.5 text-right text-slate-300 font-mono text-xs">{{ d.clean_rows }}</td>
                <td class="px-4 py-3.5 text-right text-slate-300 font-mono text-xs">{{ d.sequences }}</td>
                <td class="px-4 py-3.5 text-right text-slate-400 font-mono text-xs">{{ d.hold }}</td>
                <td class="px-4 py-3.5 text-right text-emerald-400 font-mono text-xs">{{ d.buy }}</td>
                <td class="px-4 py-3.5 text-right text-red-400 font-mono text-xs">{{ d.sell }}</td>
                <td class="px-5 py-3.5">
                  {% if d.ready %}
                  <span class="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400">
                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>Bereit
                  </span>
                  {% elif d.sequences > 0 %}
                  <span class="inline-flex items-center gap-1.5 text-xs font-medium text-amber-400">
                    <span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span>Zu wenig (min. {{ min_samples }})
                  </span>
                  {% else %}
                  <span class="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500">
                    <span class="w-1.5 h-1.5 rounded-full bg-slate-600"></span>Keine Daten
                  </span>
                  {% endif %}
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ── Tab: Daten ── -->
    <div x-show="activeTab === 'data'" x-transition:enter="transition ease-out duration-150"
         x-transition:enter-start="opacity-0 translate-y-1" x-transition:enter-end="opacity-100 translate-y-0">
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">

        <!-- Ticker -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <div class="text-xs font-bold uppercase tracking-wider text-sky-500 mb-3">Ticker</div>
          <div class="text-3xl font-bold text-slate-100">{{ stats.ticker_count }}</div>
          <div class="text-xs text-slate-500 mt-1">Einträge gesamt</div>
          {% if stats.latest_ticker_ts %}
          <div class="text-xs text-slate-600 mt-3 pt-3 border-t border-slate-800">
            Letzter: {{ stats.latest_ticker_ts | ts }}
          </div>
          {% endif %}
        </div>

        {% for tf, d in stats.timeframes.items() %}
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <div class="text-xs font-bold uppercase tracking-wider text-sky-500 mb-3">{{ tf }} Candles</div>
          <div class="text-3xl font-bold {{ 'text-emerald-400' if d.count >= 100 else ('text-amber-400' if d.count > 0 else 'text-slate-600') }}">
            {{ d.count }}
          </div>
          <div class="text-xs mt-1 {{ 'text-emerald-600' if d.count >= 100 else ('text-amber-600' if d.count > 0 else 'text-slate-600') }}">
            {% if d.count >= 100 %}Ausreichend für Training
            {% elif d.count > 0 %}Zu wenig (min. 100)
            {% else %}Noch keine Daten
            {% endif %}
          </div>
          {% if d.first_ts and d.last_ts %}
          <div class="text-xs text-slate-600 mt-3 pt-3 border-t border-slate-800 space-y-0.5">
            <div>{{ d.first_ts | ts }}</div>
            <div>→ {{ d.last_ts | ts }}</div>
          </div>
          {% endif %}
        </div>
        {% endfor %}

      </div>
    </div>

    <!-- ── Tab: Trades ── -->
    <div x-show="activeTab === 'trades'" x-transition:enter="transition ease-out duration-150"
         x-transition:enter-start="opacity-0 translate-y-1" x-transition:enter-end="opacity-100 translate-y-0">
      {% if trades %}
      <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-xs uppercase tracking-wider text-slate-500 border-b border-slate-800">
                <th class="text-left px-5 py-3">Zeit</th>
                <th class="text-left px-4 py-3">Aktion</th>
                <th class="text-right px-4 py-3">Preis</th>
                <th class="text-right px-4 py-3">Menge</th>
                <th class="text-right px-4 py-3">Gebühr</th>
                <th class="text-right px-4 py-3">PnL</th>
                <th class="text-right px-4 py-3">Portfolio</th>
                <th class="text-left px-5 py-3">Grund</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800/50">
              {% for t in trades %}
              <tr class="hover:bg-slate-800/30 transition-colors">
                <td class="px-5 py-3.5 text-slate-500 text-xs font-mono">{{ t.time }}</td>
                <td class="px-4 py-3.5">
                  <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-bold
                    {{ 'bg-emerald-950 text-emerald-400 border border-emerald-900' if t.action == 'BUY' else ('bg-red-950 text-red-400 border border-red-900' if t.action == 'SELL' else 'bg-slate-800 text-slate-400') }}">
                    {{ t.action }}
                  </span>
                </td>
                <td class="px-4 py-3.5 text-right text-slate-300 font-mono text-xs">${{ "%.4f"|format(t.price) }}</td>
                <td class="px-4 py-3.5 text-right text-slate-400 font-mono text-xs">{{ "%.4f"|format(t.quantity) }}</td>
                <td class="px-4 py-3.5 text-right text-slate-500 font-mono text-xs">${{ "%.4f"|format(t.fee) }}</td>
                <td class="px-4 py-3.5 text-right font-mono text-xs font-semibold {{ 'text-emerald-400' if t.pnl >= 0 else 'text-red-400' }}">
                  {{ '+' if t.pnl >= 0 else '' }}${{ "%.4f"|format(t.pnl) }}
                </td>
                <td class="px-4 py-3.5 text-right text-slate-300 font-mono text-xs">${{ "%.2f"|format(t.portfolio_value) }}</td>
                <td class="px-5 py-3.5 text-slate-500 text-xs">{{ t.reason or '—' }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
      {% else %}
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center">
        <div class="text-3xl mb-3">📋</div>
        <p class="text-slate-500 text-sm">Noch keine Trades vorhanden.</p>
      </div>
      {% endif %}
    </div>

    <!-- ── Tab: Simulation ── -->
    <div x-show="activeTab === 'sim'" x-transition:enter="transition ease-out duration-150"
         x-transition:enter-start="opacity-0 translate-y-1" x-transition:enter-end="opacity-100 translate-y-0">

      <!-- Controls -->
      <div class="flex flex-wrap items-center gap-3 mb-5">
        <div class="flex items-center gap-2">
          <span class="text-xs text-slate-500">Zeitrahmen:</span>
          <template x-for="tf in ['1m','5m','1h']" :key="tf">
            <button @click="sim.tf = tf"
                    :class="sim.tf === tf
                      ? 'bg-sky-900/50 text-sky-400 border-sky-700'
                      : 'bg-slate-900 text-slate-500 border-slate-800 hover:text-slate-300'"
                    class="px-3 py-1 text-xs font-semibold rounded-lg border transition-colors"
                    x-text="tf"></button>
          </template>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-xs text-slate-500">Candles:</span>
          <select x-model.number="sim.limit"
                  class="bg-slate-900 border border-slate-700 rounded-lg text-xs text-slate-300
                         px-2 py-1.5 focus:outline-none focus:border-sky-600 transition-colors">
            <option value="200">200</option>
            <option value="500" selected>500</option>
            <option value="1000">1000</option>
            <option value="2000">2000</option>
          </select>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-xs text-slate-500">Konfidenz:</span>
          <input type="range" x-model.number="sim.threshold" min="0.30" max="0.80" step="0.05"
                 class="w-24 accent-sky-400 cursor-pointer">
          <span class="text-xs font-mono font-semibold text-sky-400 w-8"
                x-text="Math.round(sim.threshold * 100) + '%'"></span>
        </div>
        <button @click="runSimulation()"
                :disabled="sim.loading"
                class="flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-xl
                       bg-emerald-900/40 text-emerald-400 border border-emerald-900
                       hover:bg-emerald-900/70 disabled:opacity-40 disabled:cursor-not-allowed transition-all">
          <span x-show="!sim.loading">&#9654; Simulation starten</span>
          <span x-show="sim.loading" class="flex items-center gap-1.5">
            <svg class="animate-spin w-3 h-3" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            Berechne…
          </span>
        </button>
        <span class="text-xs text-slate-600">LSTM-Modell auf historischen Daten</span>
      </div>

      <!-- Error state -->
      <div x-show="sim.error"
           class="bg-red-950/40 border border-red-900 rounded-2xl px-5 py-4 mb-5 text-sm text-red-400"
           x-text="sim.error"></div>

      <!-- Summary cards (shown after simulation) -->
      <div x-show="sim.data" class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 mb-5">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Start</div>
          <div class="text-xl font-bold text-slate-400" x-text="'$' + (sim.data?.summary?.start_capital ?? 0)"></div>
        </div>
        <div class="col-span-1 bg-slate-900 border rounded-2xl p-4"
             :class="(sim.data?.summary?.end_capital ?? 0) >= (sim.data?.summary?.start_capital ?? 0)
               ? 'border-emerald-900/50' : 'border-red-900/50'">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">End</div>
          <div class="text-xl font-bold"
               :class="(sim.data?.summary?.end_capital ?? 0) >= (sim.data?.summary?.start_capital ?? 0)
                 ? 'text-emerald-400' : 'text-red-400'"
               x-text="'$' + (sim.data?.summary?.end_capital?.toFixed(2) ?? 0)"></div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Return</div>
          <div class="text-xl font-bold"
               :class="(sim.data?.summary?.total_return_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'"
               x-text="((sim.data?.summary?.total_return_pct ?? 0) >= 0 ? '+' : '') + (sim.data?.summary?.total_return_pct?.toFixed(2) ?? 0) + '%'">
          </div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Trades</div>
          <div class="text-xl font-bold text-slate-300" x-text="sim.data?.summary?.trade_count ?? 0"></div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Gewonnen</div>
          <div class="text-xl font-bold text-emerald-400" x-text="sim.data?.summary?.win_count ?? 0"></div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Win Rate</div>
          <div class="text-xl font-bold"
               :class="(sim.data?.summary?.win_rate ?? 0) >= 50 ? 'text-emerald-400' : 'text-amber-400'"
               x-text="(sim.data?.summary?.win_rate ?? 0) + '%'"></div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4">
          <div class="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Candles</div>
          <div class="text-xl font-bold text-slate-400" x-text="sim.data?.summary?.candles_analyzed ?? 0"></div>
        </div>
      </div>

      <!-- Chart -->
      <div x-show="sim.data" class="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div class="flex items-center gap-4 mb-4">
          <span class="text-xs font-semibold uppercase tracking-wider text-slate-500">Kursverlauf + Trades</span>
          <span class="inline-flex items-center gap-1 text-xs text-emerald-400">
            <span class="w-2 h-2 rounded-full bg-emerald-400 inline-block"></span> Portfolio $
          </span>
          <span class="inline-flex items-center gap-1 text-xs text-sky-400">
            <span class="w-2 h-2 rounded-full bg-sky-400 inline-block"></span> Preis
          </span>
          <span class="inline-flex items-center gap-1 text-xs text-green-400">&#9650; BUY</span>
          <span class="inline-flex items-center gap-1 text-xs text-red-400">&#9660; SELL</span>
        </div>
        <div style="height: 400px; position: relative;">
          <canvas id="sim-chart"></canvas>
        </div>
      </div>

      <!-- Empty state -->
      <div x-show="!sim.data && !sim.loading && !sim.error"
           class="bg-slate-900 border border-slate-800 rounded-2xl p-16 text-center">
        <div class="text-4xl mb-4">🔬</div>
        <p class="text-slate-400 text-sm font-medium mb-2">LSTM-Simulation auf historischen Daten</p>
        <p class="text-slate-600 text-xs">Wähle Zeitrahmen und Candle-Anzahl, dann "Simulation starten" klicken.<br>
           Das Modell muss zuvor trainiert worden sein.</p>
      </div>

    </div>

  </section>

</main>

<footer class="text-center text-slate-700 text-xs py-8">
  IOTA Trading Bot &mdash; Paper Trading Only &mdash; Kein Echtgeld
</footer>

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
            df = get_recent_candles(SYMBOL, tf, limit=50000)
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
    history_json = json.dumps(metrics.get("history", {}))

    return render_template_string(
        _TEMPLATE,
        telegram_enabled=_TELEGRAM_ENABLED,
        status=status,
        trades=trades,
        signals=signals,
        position=position,
        model_ready=model_ready,
        stats=stats,
        training_stats=training_stats,
        metrics=metrics,
        history_json=history_json,
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
    notify_system("collector_start")
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
    notify_system("collector_stop")
    return jsonify({"ok": True})


# -------------------------------------------------------
# API — Historical Download
# -------------------------------------------------------
def _history_running() -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", "iota-history"],
        capture_output=True,
    )
    return result.returncode == 0


@app.route("/api/history/status")
@login_required
def api_history_status():
    return jsonify({"running": _history_running()})


@app.route("/api/history/start", methods=["POST"])
@login_required
def api_history_start():
    if _history_running():
        return jsonify({"ok": False, "message": "Download läuft bereits."})
    tf = request.args.get("timeframe", "1h")
    if tf not in ("1m", "5m", "1h"):
        return jsonify({"ok": False, "message": "Ungültiger Zeitrahmen."})
    try:
        days = int(request.args.get("days", 365))
        days = max(1, min(days, 1825))
    except ValueError:
        days = 365
    try:
        subprocess.Popen(
            ["bash", str(SCRIPT_DIR / "start_history.sh"), tf, str(days)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.error("start_history failed: %s", e)
        return jsonify({"ok": False, "message": str(e)})
    notify_system("history_start", f"{tf}, {days} Tage")
    return jsonify({"ok": True, "message": f"Download gestartet: {tf}, {days} Tage. Kann einige Minuten dauern…"})


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
    notify_system("trader_start")
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
    notify_system("trader_stop")
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
    notify_system("training_start", tf)
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
# API — ML Simulation
# -------------------------------------------------------
@app.route("/api/simulation/run", methods=["POST"])
@login_required
def api_simulation_run():
    tf = request.args.get("timeframe", "1h")
    if tf not in ("1m", "5m", "1h"):
        tf = "1h"
    try:
        limit = min(int(request.args.get("limit", 500)), 2000)
    except ValueError:
        limit = 500
    try:
        threshold = float(request.args.get("threshold", 0.45))
        threshold = max(0.30, min(threshold, 0.90))
    except ValueError:
        threshold = 0.45
    try:
        from ml_backtester import run_ml_backtest
        result = run_ml_backtest(timeframe=tf, limit=limit, threshold=threshold)
        return jsonify(result)
    except Exception as e:
        logger.error("ML simulation failed: %s", e)
        return jsonify({"error": str(e)})


# -------------------------------------------------------
# API — Telegram test
# -------------------------------------------------------
@app.route("/api/telegram/test", methods=["POST"])
@login_required
def api_telegram_test():
    if not _TELEGRAM_ENABLED:
        return jsonify({"ok": False, "message": "Telegram nicht konfiguriert (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID fehlt)."})
    ok = _tg_send("✅ *IOTA Trading Bot* — Verbindungstest erfolgreich.")
    if ok:
        return jsonify({"ok": True, "message": "Testnachricht gesendet."})
    return jsonify({"ok": False, "message": "Senden fehlgeschlagen — Token oder Chat-ID prüfen."})


# -------------------------------------------------------
# API — Chart data
# -------------------------------------------------------
@app.route("/api/chart/data")
@login_required
def api_chart_data():
    tf = request.args.get("timeframe", "1h")
    if tf not in ("1m", "5m", "1h"):
        tf = "1h"
    try:
        limit = min(int(request.args.get("limit", 200)), 1000)
    except ValueError:
        limit = 200
    df = get_recent_candles(SYMBOL, tf, limit=limit)
    if df.empty:
        return jsonify({"labels": [], "close": [], "high": [], "low": []})
    labels = [
        datetime.fromtimestamp(ts / 1000).strftime("%d.%m %H:%M")
        for ts in df["timestamp"].tolist()
    ]
    return jsonify({
        "labels": labels,
        "close": [round(float(v), 4) for v in df["close"].tolist()],
        "high":  [round(float(v), 4) for v in df["high"].tolist()],
        "low":   [round(float(v), 4) for v in df["low"].tolist()],
    })


# -------------------------------------------------------
# API — Reset portfolio
# -------------------------------------------------------
@app.route("/api/paper/reset", methods=["POST"])
@login_required
def api_paper_reset():
    try:
        count = reset_trades()
        logger.info("Portfolio reset: %d trades deleted", count)
        notify_system("portfolio_reset", f"{count} Trades gelöscht")
        return jsonify({"ok": True, "deleted": count})
    except Exception as e:
        logger.error("reset_trades failed: %s", e)
        return jsonify({"ok": False, "message": str(e)})


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
