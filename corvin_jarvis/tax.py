"""Corvin Jarvis — Tax-Aware Strategy (Tier 3.2)

한국 양도세 250만원 한도 추적 + 장기보유 우대 + tax-loss harvesting 후보.

⚠️ 일반 가이드라인 — 실제 세무는 세무사 자문 필요.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

log = logging.getLogger("corvin.tax")

# 한국 해외주식 양도세 면세 한도 (2025-2026 기준)
KR_OVERSEAS_GAIN_LIMIT_KRW = 2_500_000  # 250만원
KR_OVERSEAS_GAIN_RATE = 0.22  # 22% (지방세 포함)


def kr_capital_gain_taxable(realized_krw: int) -> dict[str, Any]:
    """한국 거주자 해외주식 실현이익 → 과세 여부.

    250만원 한도 이하: 비과세
    초과분: 22% 양도세
    """
    realized = max(0, int(realized_krw))
    taxable = max(0, realized - KR_OVERSEAS_GAIN_LIMIT_KRW)
    return {
        "realized_krw": realized,
        "limit_krw": KR_OVERSEAS_GAIN_LIMIT_KRW,
        "taxable_krw": taxable,
        "remaining_limit_krw": max(0, KR_OVERSEAS_GAIN_LIMIT_KRW - realized),
        "estimated_tax_krw": int(taxable * KR_OVERSEAS_GAIN_RATE),
    }


def long_term_holdings(
    holdings: list[dict[str, Any]],
    today: date,
    threshold_years: float = 3.0,
) -> list[dict[str, Any]]:
    """holdings 각 종목의 보유 기간 + 장기보유 flag 표시.

    holdings element는 "purchase_date" (ISO) 필드 필요.
    """
    out: list[dict[str, Any]] = []
    for h in holdings:
        sym = h["symbol"]
        pd = h.get("purchase_date")
        years_held: float | None = None
        is_long_term = False
        if pd:
            try:
                purchase = date.fromisoformat(pd) if isinstance(pd, str) else pd
                years_held = round((today - purchase).days / 365.25, 2)
                is_long_term = years_held >= threshold_years
            except (ValueError, TypeError):
                years_held = None
        out.append({
            "symbol": sym,
            "shares": h.get("shares"),
            "purchase_date": pd,
            "years_held": years_held,
            "is_long_term": is_long_term,
        })
    return out


def harvesting_candidates(
    positions: list[dict[str, Any]],
    min_loss_pct: float = -10.0,
) -> list[dict[str, Any]]:
    """미실현 손실 종목 → tax-loss harvesting 후보. loss <= min_loss_pct 인 것만."""
    out: list[dict[str, Any]] = []
    for p in positions:
        avg = float(p.get("avg_price") or 0)
        cur = float(p.get("current_price") or 0)
        if avg <= 0 or cur <= 0:
            continue
        loss_pct = (cur / avg - 1) * 100
        if loss_pct > min_loss_pct:
            continue
        out.append({
            "symbol": p["symbol"],
            "shares": p.get("shares"),
            "avg_price": avg,
            "current_price": cur,
            "unrealized_loss_pct": round(loss_pct, 2),
            "unrealized_loss_value": round((cur - avg) * float(p.get("shares") or 0), 2),
        })
    # 큰 손실 우선
    out.sort(key=lambda x: x["unrealized_loss_pct"])
    return out
