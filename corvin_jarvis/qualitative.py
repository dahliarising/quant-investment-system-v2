"""Corvin 선행 인텔리전스 — Claude 정성 분석 어댑터 (CLI 기반).

종목 사업 요약 → bull/neutral/bear verdict → 0-100.
claude CLI(-p 헤드리스)로 폐하의 Claude Code 구독 인증 사용 — ANTHROPIC_API_KEY 불필요.
CLI 미설치/실패 시 None(graceful degrade). disk_agent와 동일 패턴.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Callable

log = logging.getLogger("corvin.qualitative")

_VERDICT_SCORE = {"bull": 80.0, "neutral": 50.0, "bear": 20.0}
# CLI 출력에서 verdict 토큰 탐색 우선순위 (긴 것 우선 불필요 — 상호 배타)
_VERDICT_TOKENS = ("bull", "neutral", "bear")

_PROMPT = (
    "다음 종목의 기술 방향성·경영 의사결정·산업 포지셔닝 기준 성장 전망을 평가하라.\n"
    "종목: {symbol}\n사업: {summary}\n"
    "반드시 BULL, NEUTRAL, BEAR 중 정확히 한 단어만 출력하라. "
    "설명·근거·기타 텍스트 절대 금지. 첫 단어가 판정이다."
)


def verdict_to_score(verdict: str) -> float | None:
    """BULL/NEUTRAL/BEAR(대소문자 무관) → 점수. 그 외 None."""
    return _VERDICT_SCORE.get(verdict.strip().lower())


def extract_verdict(text: str) -> str | None:
    """텍스트에서 첫 verdict 토큰(bull/neutral/bear) 추출. CLI 잡음 대응.

    'neutral'이 'bear'보다 먼저 나오면 위치 우선. 없으면 None.
    """
    low = text.lower()
    best_pos = len(low) + 1
    best: str | None = None
    for tok in _VERDICT_TOKENS:
        pos = low.find(tok)
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best = tok
    return best


def _claude_cli_path() -> str | None:
    """claude CLI 경로. PATH 우선, 없으면 ~/.local/bin 폴백(cron PATH 대비)."""
    found = shutil.which("claude")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/claude")
    return fallback if os.path.exists(fallback) else None


def run_claude_cli(prompt: str, timeout: int = 90) -> str | None:
    """claude -p 헤드리스 호출 → stdout 텍스트. API 키 불필요(구독 인증).

    실패/미설치/타임아웃이면 None. disk_agent와 동일 패턴.
    """
    cli = _claude_cli_path()
    if cli is None:
        log.warning("claude CLI 미설치 → 정성 분석 스킵")
        return None
    try:
        proc = subprocess.run(
            [cli, "-p", "--output-format", "text"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout or "").strip()
        return out or None
    except Exception as e:  # noqa: BLE001
        log.warning("claude CLI 호출 실패: %s", e)
        return None


def qualitative_score_via_cli(
    symbol: str,
    summary: str,
    runner: Callable[[str], str | None] | None = None,
) -> float | None:
    """Claude CLI(구독)로 정성 verdict → 점수. runner 주입 시 테스트(미호출).

    API 키 대신 폐하의 Claude Code 인증을 사용. 출력 잡음은 extract_verdict로 흡수.
    """
    run = runner or run_claude_cli
    prompt = _PROMPT.format(symbol=symbol, summary=summary)
    out = run(prompt)
    if not out:
        return None
    verdict = extract_verdict(out)
    return verdict_to_score(verdict) if verdict else None
