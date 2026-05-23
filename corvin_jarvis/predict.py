"""Corvin Jarvis — Predictive Alerts (Tier 2.3)

시계열 DB + log-normal 가정으로 미래 N일 가격 임계 이탈 확률 계산.

⚠️ 통계 추정치 — 실제 매매 결정의 근거가 아닌 advisory only.
"""
from __future__ import annotations

import logging
import math
import statistics
from pathlib import Path
from typing import Any

from corvin_jarvis import timeseries

log = logging.getLogger("corvin.predict")


def log_returns(db_path: Path, symbol: str, days: int = 60) -> list[float]:
    """timeseries.db에서 symbol의 N일치 가격을 시간순으로 가져와 log returns."""
    rows = timeseries.read_history(db_path, symbol=symbol, limit=days * 24)
    prices = [r["price"] for r in rows if r["price"] is not None and r["price"] > 0]
    if len(prices) < 2:
        return []
    out: list[float] = []
    for prev, cur in zip(prices[:-1], prices[1:]):
        if prev > 0 and cur > 0:
            out.append(math.log(cur / prev))
    return out


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def probability_below(
    current_price: float,
    threshold: float,
    mu: float,
    sigma: float,
    horizon_days: int,
) -> float:
    """P(S_N < threshold | S_0 = current_price), log-normal endpoint approx.

    log(S_N) ~ Normal(log(S_0) + N*mu, N*sigma^2).
    sigma=0 degenerate → step function.
    """
    if sigma <= 0:
        return 1.0 if threshold >= current_price else 0.0
    if current_price <= 0 or threshold <= 0:
        return 0.0
    log_ratio = math.log(threshold / current_price)
    drift = horizon_days * mu
    vol = sigma * math.sqrt(horizon_days)
    z = (log_ratio - drift) / vol
    return round(_norm_cdf(z), 6)


def _severity_from_prob(p: float) -> str:
    if p >= 0.5:
        return "critical"
    if p >= 0.3:
        return "high"
    if p >= 0.15:
        return "medium"
    return "low"


def build_predictive_alerts(
    db_path: Path,
    levels: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """levels = {symbol: {threshold, horizon_days, min_prob}}.

    각 symbol의 log-return mean/std로 lognormal endpoint P 계산 후
    min_prob 이상이면 alert 생성.
    """
    alerts: list[dict[str, Any]] = []
    for symbol, spec in levels.items():
        threshold = float(spec["threshold"])
        horizon = int(spec["horizon_days"])
        min_prob = float(spec.get("min_prob", 0.1))
        returns = log_returns(db_path, symbol, days=60)
        if len(returns) < 3:
            continue
        latest_history = timeseries.read_history(db_path, symbol=symbol, limit=1)
        if not latest_history or latest_history[-1]["price"] is None:
            continue
        current = float(latest_history[-1]["price"])
        mu = statistics.fmean(returns)
        try:
            sigma = statistics.stdev(returns)
        except statistics.StatisticsError:
            sigma = 0.0
        prob = probability_below(current, threshold, mu=mu, sigma=sigma, horizon_days=horizon)
        if prob < min_prob:
            continue
        direction = "이탈" if threshold < current else "도달"
        alerts.append({
            "category": "predictive",
            "metric": symbol,
            "severity": _severity_from_prob(prob),
            "message": (
                f"{symbol} ${threshold:.2f} {direction} 확률 "
                f"{prob*100:.1f}% (D+{horizon}d, current=${current:.2f}, "
                f"σ={sigma*100:.2f}%/d, n={len(returns)})"
            ),
            "value": round(prob, 4),
            "threshold": min_prob,
            "delta_from_prev": None,
        })
    return alerts
