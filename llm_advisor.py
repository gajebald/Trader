import json
import logging

import requests

from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

_DEFAULT_HOLD = {
    "decision": "HOLD",
    "confidence": 0.0,
    "reason": "llm_unavailable",
    "risk_level": "HIGH",
}

_VALID_DECISIONS = {"BUY", "SELL", "HOLD"}
_VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


def _build_prompt(signals: dict, position: dict | None) -> str:
    pos_info = "None (no open position)"
    if position:
        pos_info = (
            f"OPEN BUY at entry price {position.get('price', 'N/A'):.4f} USD, "
            f"quantity {position.get('quantity', 'N/A'):.4f} IOTA"
        )

    lines = [
        "You are a professional crypto trading assistant. Analyze the following IOTA/USD market data",
        "and provide a structured trading recommendation.",
        "",
        "=== Market Data ===",
        f"Last Price : {signals.get('last_price', 'N/A')}",
        f"High (bar) : {signals.get('high', 'N/A')}",
        f"Low  (bar) : {signals.get('low', 'N/A')}",
        f"Volume     : {signals.get('volume', 'N/A')}",
        f"Pct Change : {signals.get('pct_change', 'N/A')}",
        "",
        "=== Technical Indicators ===",
        f"SMA 20      : {signals.get('sma_20', 'N/A')}",
        f"SMA 50      : {signals.get('sma_50', 'N/A')}",
        f"EMA 20      : {signals.get('ema_20', 'N/A')}",
        f"RSI 14      : {signals.get('rsi', 'N/A')}",
        f"MACD        : {signals.get('macd', 'N/A')}",
        f"MACD Signal : {signals.get('macd_signal', 'N/A')}",
        f"MACD Hist   : {signals.get('macd_hist', 'N/A')}",
        f"Volatility  : {signals.get('volatility', 'N/A')}",
        "",
        "=== Current Position ===",
        pos_info,
        "",
        "=== Instructions ===",
        "Based on this data, decide whether to BUY, SELL, or HOLD IOTA.",
        "Consider trend direction (SMA/EMA), momentum (MACD), RSI levels, and current position.",
        "",
        'Respond ONLY with valid JSON, no other text:',
        '{"decision": "BUY" | "SELL" | "HOLD", "confidence": 0.0-1.0, '
        '"reason": "short explanation", "risk_level": "LOW" | "MEDIUM" | "HIGH"}',
    ]
    return "\n".join(lines)


def _parse_llm_response(raw_text: str) -> dict:
    try:
        text = raw_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        parsed = json.loads(text[start:end])

        decision = str(parsed.get("decision", "HOLD")).upper()
        if decision not in _VALID_DECISIONS:
            logger.warning("LLM returned unknown decision '%s', defaulting to HOLD", decision)
            decision = "HOLD"

        risk = str(parsed.get("risk_level", "HIGH")).upper()
        if risk not in _VALID_RISK_LEVELS:
            risk = "HIGH"

        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": str(parsed.get("reason", ""))[:200],
            "risk_level": risk,
        }
    except Exception as e:
        logger.warning("Failed to parse LLM response: %s | raw: %.200s", e, raw_text)
        return {**_DEFAULT_HOLD, "reason": "parse_error"}


def get_advice(signals: dict, position: dict | None = None) -> dict:
    prompt = _build_prompt(signals, position)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("response", "")
        advice = _parse_llm_response(raw_text)
        logger.info(
            "LLM advice: decision=%s confidence=%.2f risk=%s reason=%s",
            advice["decision"], advice["confidence"], advice["risk_level"], advice["reason"],
        )
        return advice
    except requests.exceptions.RequestException as e:
        logger.warning("Ollama unavailable: %s — defaulting to HOLD", e)
        return {**_DEFAULT_HOLD}
