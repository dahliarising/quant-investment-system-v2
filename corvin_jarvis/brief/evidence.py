"""신호의 검증 근거 스택. 표본 부족 시 '검증부족' 정직 라벨 (거짓 확신 금지)."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.types import EvidenceStack


def build_evidence(calibration: dict[str, Any], *, engine: str, kind: str,
                   min_samples: int = 10) -> EvidenceStack:
    entry = (calibration.get(engine) or {}).get(kind)
    if not entry:
        return EvidenceStack(validation="검증부족", edge_summary=None)
    n = int(entry.get("n", 0))
    if n < min_samples:
        return EvidenceStack(validation=f"검증부족(n={n})", edge_summary=None)
    hit = float(entry.get("hit_rate", 0.0)) * 100
    return EvidenceStack(validation=f"n={n}·적중률 {hit:.0f}%", edge_summary=None)
