"""Corvin 선행 인텔리전스 — 오케스트레이터.

4 Pillar가 반환한 LeadingSignal을 수집 → 신뢰도 게이트 → 한국어 브리프 포맷.
Plan 1은 Pillar 2(event_calendar)만 연결. 후속 plan에서 Pillar 1/3/4 추가.

전송(channels.send_telegram)은 기존 jarvis 파이프라인이 담당.
이 모듈은 "보낼 문자열"을 반환하는 데까지만 책임진다.
"""
from __future__ import annotations

from corvin_jarvis.signals.leading_signal import (
    LeadingSignal,
    apply_confidence_gate,
)


def format_brief(signals: list[LeadingSignal], threshold: float = 60.0) -> str:
    """게이트 통과 신호를 한국어 브리프 문자열로 포맷.

    - advisory 신호는 "참고용" 태그를 붙여 본문 하단에 한 줄로만.
    - 통과 신호가 없으면 빈 문자열 반환(= 전송 안 함).
    """
    passed = apply_confidence_gate(signals, threshold)
    if not passed:
        return ""

    primary = [s for s in passed if not s.advisory]
    advisory = [s for s in passed if s.advisory]

    lines: list[str] = ["**🔭 Corvin 선행 인텔리전스 브리프**", ""]
    for s in primary:
        lines.append(f"- {s.message} (신뢰도 {s.confidence:.0f})")

    if advisory:
        lines.append("")
        lines.append("_참고용 (검증 약함):_")
        for s in advisory:
            lines.append(f"- {s.message}")

    return "\n".join(lines)
