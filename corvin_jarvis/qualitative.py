"""Corvin 선행 인텔리전스 — Claude haiku 정성 분석 어댑터.

종목 사업 요약 → bull/neutral/bear verdict → 0-100. DI client로 테스트.
키/클라이언트 없으면 None(graceful degrade). 모델: claude-haiku-4-5 (저비용).
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("corvin.qualitative")

MODEL = "claude-haiku-4-5"
_VERDICT_SCORE = {"bull": 80.0, "neutral": 50.0, "bear": 20.0}

_PROMPT = (
    "다음 종목의 기술 방향성·경영 의사결정·산업 포지셔닝을 평가해 "
    "성장 전망을 한 단어로만 답하라: BULL, NEUTRAL, BEAR 중 하나.\n"
    "종목: {symbol}\n사업: {summary}\n답(한 단어):"
)


def verdict_to_score(verdict: str) -> float | None:
    """BULL/NEUTRAL/BEAR(대소문자 무관) → 점수. 그 외 None."""
    return _VERDICT_SCORE.get(verdict.strip().lower())


def default_client() -> Any | None:
    """ANTHROPIC_API_KEY 있으면 anthropic 클라이언트, 없으면 None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception as e:  # noqa: BLE001
        log.warning("anthropic client 생성 실패: %s", e)
        return None


def qualitative_score(
    symbol: str, summary: str, client: Any | None
) -> float | None:
    """Claude haiku로 정성 verdict → 점수. client None이면 None(degrade)."""
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user",
                       "content": _PROMPT.format(symbol=symbol, summary=summary)}],
        )
        text = msg.content[0].text if msg.content else ""
        return verdict_to_score(text)
    except Exception as e:  # noqa: BLE001
        log.warning("정성 분석 실패 %s: %s", symbol, e)
        return None
