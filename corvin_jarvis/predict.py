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
