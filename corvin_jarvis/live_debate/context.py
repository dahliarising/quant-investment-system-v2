"""Phase A — 검증 컨텍스트 빌더.

토론 에이전트가 공유할 '검증된 사실 묶음'. 모든 수치는 data_verify로 ≥2소스
교차검증되고, EW 신호는 백테스트 신뢰도로 태깅된다. 여기 없는 수치는 토론에
인용될 수 없다(fact-check 기준). 컨텍스트는 LLM이 아니라 *코드*가 만든다.
"""
from __future__ import annotations

from typing import Callable

from corvin_jarvis import data_verify as dv


def build_context(holdings: list[dict],
                  fetchers_for: Callable[[str], dict[str, Callable[[], float | None]]],
                  tol_pct: float = 1.5) -> dict:
    """보유 종목 가격을 ≥2소스 교차검증, 손익은 저장vs라이브 정합.

    holdings: [{symbol, shares, avg_price, stored_pnl_pct}]
    fetchers_for(sym): {source_name: ()->price}
    """
    enriched = []
    low_conf = []
    for h in holdings:
        sym = h["symbol"]
        price = dv.verified(sym, fetchers_for(sym), tol_pct=tol_pct, positive=True)
        live_pnl = None
        if price["value"] is not None and h.get("avg_price"):
            live_pnl = (price["value"] / h["avg_price"] - 1) * 100
        pnl = dv.reconcile_pnl(h.get("stored_pnl_pct"), live_pnl)
        conf = price["confidence"]
        if conf in ("low", "none"):
            low_conf.append(sym)
        enriched.append({
            "symbol": sym,
            "shares": h.get("shares"),
            "avg_price": h.get("avg_price"),
            "price": price,
            "live_pnl_pct": live_pnl,
            "pnl": pnl,
            "confidence": conf,
        })
    return {"holdings": enriched, "low_confidence": low_conf,
            "n_verified": sum(1 for e in enriched if e["confidence"] == "high")}


def attach_signal_reliability(ctx: dict, backtest_report: dict) -> dict:
    """EW 신호별 백테스트 신뢰도 태깅 — 약한 신호 과신 차단.

    정밀도가 기저율의 ~1.5배 이상이면 'has-edge', 아니면 'low'.
    """
    rel = {}
    for key in ("semis", "vix_term", "breadth", "hy", "curve"):
        r = backtest_report.get(f"{key}_red")
        if not r:
            continue
        base = r.get("n_events_rate", 0.0)
        prec = r.get("precision", 0.0)
        edge = "has-edge" if base and prec >= base * 1.5 else "low"
        rel[key] = {"edge": edge, "precision": prec, "base_rate": base}
    return {**ctx, "signal_reliability": rel}


def context_facts(ctx: dict) -> set:
    """fact-check가 대조할 '허용된 사실' 집합 (symbol, field, value)."""
    facts: set = set()
    for h in ctx.get("holdings", []):
        sym = h["symbol"]
        if h.get("live_pnl_pct") is not None:
            facts.add((sym, "live_pnl_pct", round(h["live_pnl_pct"], 2)))
        price = h.get("price", {})
        if price.get("value") is not None:
            facts.add((sym, "price", round(price["value"], 2)))
    return facts
