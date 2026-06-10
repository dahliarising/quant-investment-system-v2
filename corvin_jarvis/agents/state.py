"""TradingState — LangGraph 상태 컨테이너."""
from __future__ import annotations

from typing import Any, TypedDict


class TradingState(TypedDict):
    symbol: str
    latest: dict[str, Any]       # pulse.py latest.json
    alerts: list[dict[str, Any]] # compare.py alerts

    # 에이전트 출력
    bull_thesis: str
    bear_thesis: str
    research_summary: str
    confidence: float            # 0.0 ~ 1.0

    # 최종 판정
    action: str                  # "buy" | "sell" | "hold"
    rationale: str
