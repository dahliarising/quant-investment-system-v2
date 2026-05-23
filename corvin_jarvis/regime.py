"""Corvin Jarvis — Regime Detector (Tier 2.2)

VIX + USD/KRW + (optional) yield spread / correlation → risk-on/neutral/risk-off/crisis 라벨.

매 pulse 후 호출. 라벨 전환 시 alert 자동 생성.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import statistics
from datetime import date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.regime")

LABELS = ("crisis", "risk_off", "neutral", "risk_on", "euphoria")


def label_from_signals(
    vix: float | None,
    vix_zscore: float | None,
    usd_krw_pct: float | None,
    correlation: float | None = None,
) -> dict[str, Any]:
    """순수 함수: 시그널 → label + score(-100~100) + drivers.

    Score 합산 방식 (각 -100~+100 기여):
    - VIX 절대값: 낮을수록 risk-on. 12 미만 +40 / 20 이상 -20 / 30 이상 -60 / 45 이상 -100
    - VIX zscore: 양수면 risk-off 가중치
    - USD/KRW pct: 1% 이상 약달러 vs 원약세 → risk-off
    - correlation: KOSPI-SPY 디커플링 (음수) → mixed risk
    """
    score = 0.0
    drivers: list[str] = []

    if vix is not None:
        if vix < 13:
            score += 30
            drivers.append(f"VIX {vix:.1f} (low, complacency)")
        elif vix < 18:
            score += 10
        elif vix < 25:
            drivers.append(f"VIX {vix:.1f} (elevated)")
        elif vix < 35:
            score -= 25
            drivers.append(f"VIX {vix:.1f} (high stress)")
        else:
            score -= 60
            drivers.append(f"VIX {vix:.1f} (crisis level)")

    if vix_zscore is not None:
        # zscore > 2 → spike → -20; < -1 → calm → +10
        score += max(-20.0, min(10.0, -10.0 * vix_zscore))
        if vix_zscore >= 2.0:
            drivers.append(f"VIX Z={vix_zscore:+.2f}σ (spike)")

    if usd_krw_pct is not None:
        # 원화 약세 (양수) = risk-off pressure
        if abs(usd_krw_pct) >= 2.0:
            score -= 25
            drivers.append(f"USD/KRW {usd_krw_pct:+.2f}% (FX stress)")
        elif abs(usd_krw_pct) >= 1.0:
            score -= 10
            drivers.append(f"USD/KRW {usd_krw_pct:+.2f}% (moderate FX move)")

    if correlation is not None and correlation < 0.0:
        score -= 10
        drivers.append(f"KOSPI-SPY decoupled (ρ={correlation:.2f})")

    score = max(-100.0, min(100.0, score))

    if score <= -60:
        label = "crisis"
    elif score <= -30:
        label = "risk_off"
    elif score < 30:
        label = "neutral"
    elif score < 60:
        label = "risk_on"
    else:
        label = "euphoria"

    return {
        "label": label,
        "score": round(score, 1),
        "drivers": drivers,
    }
