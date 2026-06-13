"""설득 브리핑 컴포넌트 간 frozen dataclass 인터페이스."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionLine:
    symbol: str
    pnl_pct: float | None
    action_icon: str          # ✂️ ✅ ➕ 👀
    label: str                # 종목당 한 줄 액션
    fresh_label: str          # "종가"/"전일종가" 등 가격 신선도


@dataclass(frozen=True)
class EvidenceStack:
    validation: str           # "n=12·적중률 58%" 또는 "검증부족"
    edge_summary: str | None  # 백테스트 엣지 요약 (E1엔 보통 None)


@dataclass(frozen=True)
class Framing:
    bull: str
    bear: str
    base: str
    counterfactual: str


@dataclass(frozen=True)
class PsychGuard:
    triggered: bool
    pattern: str | None       # 항복매수/사건전FOMO/드로다운/앵커링
    question: str | None      # "룰 진입 vs 감정 반응?" 자문 1줄


@dataclass(frozen=True)
class Brief:
    headline: str
    positions: list[PositionLine]
    evidence: dict[str, EvidenceStack]
    framing: Framing | None
    psych: PsychGuard | None
    as_of: str
    market_state: str
