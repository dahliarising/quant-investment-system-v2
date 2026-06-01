"""Corvin 선행 인텔리전스 — 모든 Pillar 공통 신호 계약.

각 Pillar(fundamental/event/ensemble/cross_market)는 이 LeadingSignal을 반환한다.
오케스트레이터는 Pillar 종류와 무관하게 게이트·포맷·디스패치한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_PILLARS = {"fundamental", "event", "ensemble", "cross_market"}
VALID_DIRECTIONS = {"bull", "neutral", "bear"}
VALID_HORIZONS = {"intraday", "days", "weeks"}


@dataclass(frozen=True)
class LeadingSignal:
    """선행 신호 단일 계약. frozen → 불변(immutability 원칙)."""

    pillar: str
    symbol: str
    direction: str          # bull | neutral | bear
    confidence: float       # 0-100
    score: float | None     # pillar별 원점수 (예: Growth Score). 없으면 None
    horizon: str            # intraday | days | weeks
    advisory: bool          # True면 "참고용" 태그 강제 (검증 약한 Pillar)
    message: str            # 한국어 해석
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pillar not in VALID_PILLARS:
            raise ValueError(f"invalid pillar: {self.pillar}")
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(f"invalid direction: {self.direction}")
        if self.horizon not in VALID_HORIZONS:
            raise ValueError(f"invalid horizon: {self.horizon}")
        if not (0.0 <= self.confidence <= 100.0):
            raise ValueError(f"confidence out of range: {self.confidence}")
