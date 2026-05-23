"""Corvin Jarvis — Rebalancing Engine (Tier 2.5)

목표 비중 (config) vs 실제 → drift 감지 → 매수/매도 제안 (advisory only).
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("corvin.rebalance")

_DEFAULT_USD_KRW = 1500.0


def _portfolio_values(
    holdings: list[dict[str, Any]],
    prices: dict[str, float],
    usd_krw: float,
) -> tuple[dict[str, float], float]:
    """holdings 별 USD-equivalent value + total."""
    values: dict[str, float] = {}
    for h in holdings:
        sym = h["symbol"]
        if sym not in prices:
            continue
        shares = float(h["shares"])
        cur = h.get("currency", "USD")
        local = shares * prices[sym]
        values[sym] = local / usd_krw if cur == "KRW" else local
    return values, sum(values.values())


def detect_drift(
    holdings: list[dict[str, Any]],
    target_weights: dict[str, float],
    market_prices: dict[str, float],
    tolerance: float = 0.05,
    usd_krw: float = _DEFAULT_USD_KRW,
) -> list[dict[str, Any]]:
    """target_weights (0~1) vs 실제 비중. drift > tolerance면 entry 생성."""
    values, total = _portfolio_values(holdings, market_prices, usd_krw)
    if total <= 0:
        return []
    drifts: list[dict[str, Any]] = []
    for sym, target in target_weights.items():
        actual_w = values.get(sym, 0.0) / total
        drift_pp = (actual_w - target) * 100
        if abs(drift_pp) < tolerance * 100:
            continue
        direction = "overweight" if drift_pp > 0 else "underweight"
        drifts.append({
            "symbol": sym,
            "target_pct": round(target * 100, 2),
            "actual_pct": round(actual_w * 100, 2),
            "drift_pp": round(drift_pp, 2),
            "direction": direction,
        })
    return drifts


def propose_trades(
    holdings: list[dict[str, Any]],
    target_weights: dict[str, float],
    market_prices: dict[str, float],
    tolerance: float = 0.05,
    usd_krw: float = _DEFAULT_USD_KRW,
) -> list[dict[str, Any]]:
    """drift entries → 매수/매도 shares 제안 (advisory only)."""
    drifts = detect_drift(holdings, target_weights, market_prices, tolerance, usd_krw)
    if not drifts:
        return []
    _, total = _portfolio_values(holdings, market_prices, usd_krw)
    trades: list[dict[str, Any]] = []
    for d in drifts:
        sym = d["symbol"]
        if sym not in market_prices:
            continue
        target_value = (d["target_pct"] / 100) * total
        actual_value = (d["actual_pct"] / 100) * total
        diff = target_value - actual_value  # +면 매수, -면 매도
        price = market_prices[sym]
        shares_needed = abs(diff) / price
        trades.append({
            "symbol": sym,
            "side": "buy" if diff > 0 else "sell",
            "shares": round(shares_needed, 4),
            "approx_value_usd": round(abs(diff), 2),
            "current_drift_pp": d["drift_pp"],
        })
    return trades
