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
