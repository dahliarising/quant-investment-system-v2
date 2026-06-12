"""투자심리 가드 — 4대 패턴 트리거 시 '룰 vs 감정' 자문 1줄."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.types import PsychGuard

_DRAWDOWN_PCT = -12.0


def build_psych_guard(positions: list[dict[str, Any]], *,
                      has_new_buy_candidate: bool) -> PsychGuard:
    deep = [p for p in positions
            if p.get("pnl_pct") is not None and p["pnl_pct"] <= _DRAWDOWN_PCT]
    if deep:
        return PsychGuard(
            triggered=True, pattern="드로다운",
            question="드로다운에 흔들리는 중? 룰 손절선 vs 감정 반응 먼저 점검.",
        )
    if has_new_buy_candidate:
        return PsychGuard(
            triggered=True, pattern="FOMO",
            question="추격 진입인가? 룰 진입존 도달 vs FOMO 자문.",
        )
    return PsychGuard(triggered=False, pattern=None, question=None)
