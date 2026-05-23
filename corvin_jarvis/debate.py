"""Corvin Jarvis — Multi-Agent Debate Context Layer (Tier 4.1)

Bull/Bear 각 측 retrieval. LLM 합성(actual debate)은 runtime에 Corvin 본인이 수행.

이 모듈은 데이터 구조화만 — context만 풍부하게 모아주면 Bull/Bear LLM call이
효과적인 토론 가능.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from corvin_jarvis import qa, timeseries

log = logging.getLogger("corvin.debate")


def _momentum_signal(prices: list[float]) -> str | None:
    """단순 추세: 최근 절반 평균 vs 직전 절반 평균."""
    if len(prices) < 4:
        return None
    half = len(prices) // 2
    early = sum(prices[:half]) / half
    late = sum(prices[half:]) / (len(prices) - half)
    if late > early * 1.005:
        return "upward momentum"
    if late < early * 0.995:
        return "downward momentum"
    return None


def gather_bull_context(
    db_path: Path,
    symbol: str,
    benchmark: str = "sp500",
    days: int = 7,
) -> dict[str, Any]:
    """매수/보유 옹호 신호 모음 (raw, LLM 합성용)."""
    signals: list[str] = []
    hist = qa.recent_history_summary(db_path, symbol=symbol, days=days)
    if hist["n"] == 0:
        return {"symbol": symbol, "signals": []}

    rel = qa.relative_strength(db_path, symbol=symbol, benchmark=benchmark, days=days)
    if rel["verdict"] == "outperform":
        signals.append(
            f"벤치마크 outperform ({rel['symbol_pct']:+.2f}% vs {benchmark} "
            f"{rel['benchmark_pct']:+.2f}%, 상대 {rel['relative_pp']:+.2f}pp)"
        )

    rows = timeseries.read_history(db_path, symbol=symbol, limit=days * 24)
    prices = [r["price"] for r in rows if r["price"] is not None]
    mom = _momentum_signal(prices)
    if mom == "upward momentum":
        signals.append("upward momentum — 최근 절반 평균이 직전 절반 평균 대비 +0.5% 이상")

    if hist["pct_change"] is not None and hist["pct_change"] > 0:
        signals.append(f"{days}일 누적 {hist['pct_change']:+.2f}% 상승")

    if hist["end_price"] is not None and hist["max_price"] is not None:
        if hist["end_price"] >= hist["max_price"] * 0.98:
            signals.append(f"고점 근처 (${hist['end_price']:.2f}, max ${hist['max_price']:.2f})")

    return {"symbol": symbol, "signals": signals, "history": hist, "relative": rel}


def gather_bear_context(
    db_path: Path,
    symbol: str,
    benchmark: str = "sp500",
    days: int = 7,
) -> dict[str, Any]:
    """매도/관망 옹호 신호 모음."""
    signals: list[str] = []
    hist = qa.recent_history_summary(db_path, symbol=symbol, days=days)
    if hist["n"] == 0:
        return {"symbol": symbol, "signals": []}

    rel = qa.relative_strength(db_path, symbol=symbol, benchmark=benchmark, days=days)
    if rel["verdict"] == "underperform":
        signals.append(
            f"벤치마크 underperform ({rel['symbol_pct']:+.2f}% vs {benchmark} "
            f"{rel['benchmark_pct']:+.2f}%, 상대 {rel['relative_pp']:+.2f}pp)"
        )

    rows = timeseries.read_history(db_path, symbol=symbol, limit=days * 24)
    prices = [r["price"] for r in rows if r["price"] is not None]
    mom = _momentum_signal(prices)
    if mom == "downward momentum":
        signals.append("downward momentum — 최근 절반 평균이 직전 절반 평균 대비 -0.5% 이상")

    if hist["pct_change"] is not None and hist["pct_change"] < 0:
        signals.append(f"{days}일 누적 {hist['pct_change']:+.2f}% 하락")

    if hist["end_price"] is not None and hist["min_price"] is not None:
        if hist["end_price"] <= hist["min_price"] * 1.02:
            signals.append(f"저점 근처 (${hist['end_price']:.2f}, min ${hist['min_price']:.2f})")

    return {"symbol": symbol, "signals": signals, "history": hist, "relative": rel}


def debate_brief(
    db_path: Path,
    symbol: str,
    benchmark: str = "sp500",
    days: int = 7,
) -> dict[str, Any]:
    """Bull + Bear context를 하나의 dict로. Corvin LLM이 prompts로 전달 후 합성."""
    return {
        "symbol": symbol,
        "bull": gather_bull_context(db_path, symbol, benchmark, days),
        "bear": gather_bear_context(db_path, symbol, benchmark, days),
    }
