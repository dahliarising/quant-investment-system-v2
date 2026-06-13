"""모든 예측 모듈이 반환하는 단일 출력 계약.

data_ok=False면 다이제스트에서 '⏸ 보류'로 축약 — 가짜 정밀도 방지(Corvin 철칙).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionResult:
    system: str                 # velocity|probability|momentum|geopolitical|vector_analog
    scope: str                  # "market" 또는 종목 심볼
    verdict: str                # 한 줄 결론 (한국어)
    confidence: float           # 0-100
    evidence: dict[str, Any] = field(default_factory=dict)
    data_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def insufficient(system: str, scope: str, reason: str) -> PredictionResult:
    """데이터 부족 시 표준 '보류' 결과."""
    return PredictionResult(system=system, scope=scope,
                            verdict="데이터 부족 · 보류", confidence=0.0,
                            evidence={"reason": reason}, data_ok=False)
