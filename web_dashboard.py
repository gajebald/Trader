"""
Web-Dashboard für den IOTA Trading Bot.
Zeigt Portfolio, Trades und aktuelle Marktsignale.
Läuft auf localhost, nginx proxied von außen.
"""
import os
import logging
from datetime import datetime

from flask import Flask, render_template_string

from config import (
    SYMBOL, STARTING_CAPITAL, DASHBOARD_HOST, DASHBOARD_PORT,
    DASHBOARD_REFRESH_SECONDS, MODEL_PATH,
)
from database import get_all_trades, get_open_position, get_latest_ticker, get_recent_candles, get_data_stats
from indicators import calculate_all, get_latest_signals
from paper_trader import get_portfolio_status

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.template_filter("ts")
def _ts_filter(ms):
    """Formatiert einen Unix-Millisekunden-Timestamp als lesbares Datum."""
    if not ms:
        return "—"
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%d.%m.%y %H:%M")
    except Exception:
        return "—"

# -------------------------------------------------------
# HTML-Template (inline, keine externen Abhängigkeiten)
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
    h1 { font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .subtitle { color: #64748b; font-size: 0.85rem; margin-bottom: 28px; }
    .subtitle .updated { color: #38bdf8; }

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
    .dot-yellow { background: #facc15; }

    .empty { color: #475569; font-style: italic; text-align: center; padding: 20px; }
    footer { text-align: center; color: #334155; font-size: 0.75rem; margin-top: 12px; }

    .tf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
    .tf-card { background: #161b27; border-radius: 8px; padding: 12px 14px; }
    .tf-name { font-size: 0.75rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
    .tf-count { font-size: 1.4rem; font-weight: 700; color: #f8fafc; }
    .tf-label { font-size: 0.7rem; color: #64748b; margin-top: 2px; }
    .tf-range { font-size: 0.72rem; color: #475569; margin-top: 6px; border-top: 1px solid #1e2330; padding-top: 6px; }
  </style>
</head>
<body>

<h1>📈 IOTA Trading Bot</h1>
<p class="subtitle">
  Symbol: <strong>{{ symbol }}</strong> &nbsp;|&nbsp;
  Aktualisierung alle {{ refresh }}s &nbsp;|&nbsp;
  <span class="updated">{{ now }}</span>
</p>

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
  <div class="empty">Noch keine Signaldaten — bitte zuerst 'python main.py collect' ausführen.</div>
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


def _format_trades(raw: list) -> list:
    result = []
    for t in reversed(raw[-20:]):
        try:
            ts = datetime.fromtimestamp(t["timestamp"] / 1000).strftime("%d.%m %H:%M")
        except Exception:
            ts = "—"
        result.append({**t, "time": ts})
    return result


# -------------------------------------------------------
# Route
# -------------------------------------------------------
@app.route("/")
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

    return render_template_string(
        _TEMPLATE,
        status=status,
        trades=trades,
        signals=signals,
        position=position,
        model_ready=model_ready,
        stats=stats,
        symbol=SYMBOL,
        start_capital=STARTING_CAPITAL,
        refresh=DASHBOARD_REFRESH_SECONDS,
        now=datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
    )


# -------------------------------------------------------
# Einstiegspunkt
# -------------------------------------------------------
def run():
    from database import setup_database
    setup_database()
    logger.info("Dashboard startet auf http://%s:%d", DASHBOARD_HOST, DASHBOARD_PORT)
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run()
