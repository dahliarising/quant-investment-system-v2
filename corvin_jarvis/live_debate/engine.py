"""Phase B — 토론 엔진.

페르소나 × 라운드(opening→rebuttal→synthesis) 오케스트레이션. LLM 호출은 DI.
debate()는 턴을 yield → 실시간 스트리밍의 토대. 순수 오케스트레이션(IO 없음).
"""
from __future__ import annotations

import json
from typing import Callable, Iterator

from corvin_jarvis.live_debate import personas as _p

_ROUND_INSTRUCTION = {
    "opening": "공유 데이터를 근거로 너의 방법론에 따른 입장을 카톡처럼 짧게 3~4 말풍선. 전문용어는 괄호 풀이.",
    "rebuttal": "다른 참가자들의 직전 주장에 반박하는 카톡 말풍선 2~3개.",
    "synthesis": "지금까지 토론을 너의 관점에서 한 줄로 정리.",
    "reply": "폐하의 질문/의견에 너의 방법론 관점으로 짧게(1~2 말풍선) 직접 답하라.",
}


def build_prompt(persona: dict, context: dict, round_type: str,
                 prior_turns: list[dict] | None = None) -> str:
    parts = [
        f"[너의 페르소나] {persona['system']}",
        f"[공유 검증 데이터] {json.dumps(context, ensure_ascii=False, default=str)}",
    ]
    if prior_turns:
        prev = " / ".join(f"{t['name']}: {t['text']}" for t in prior_turns)
        parts.append(f"[직전 발언들] {prev}")
    parts.append("[지시] " + _ROUND_INSTRUCTION.get(round_type, ""))
    parts.append("데이터에 없는 수치는 절대 지어내지 마라(추측 금지). 원문 텍스트만 반환.")
    return "\n".join(parts)


def run_round(persona_list: list[dict], context: dict, round_type: str,
              llm: Callable[[str], str],
              prior_turns: list[dict] | None = None) -> list[dict]:
    """한 라운드 — 페르소나마다 LLM 1회 호출 → 턴 리스트."""
    turns = []
    for p in persona_list:
        text = llm(build_prompt(p, context, round_type, prior_turns))
        turns.append({
            "persona": p["id"], "name": p["name"], "tag": p.get("tag"),
            "avatar": p["avatar"], "accent": p["accent"],
            "credibility": p.get("credibility"), "cred_note": p.get("cred_note"),
            "round": round_type, "text": text,
        })
    return turns


def debate(mode: str, context: dict, llm: Callable[[str], str],
           rounds: list[str] | None = None) -> Iterator[dict]:
    """라운드 순차 실행 — 턴을 yield(스트리밍 토대)."""
    persona_list = _p.for_mode(mode)
    rounds = rounds or ["opening", "rebuttal", "synthesis"]
    history: list[dict] = []
    for rt in rounds:
        prior = list(history) if rt != "opening" else None
        for turn in run_round(persona_list, context, rt, llm, prior):
            history.append(turn)
            yield turn


def respond_to_user(mode: str, context: dict, question: str,
                    llm: Callable[[str], str]) -> Iterator[dict]:
    """폐하의 질문/의견에 각 페르소나가 답하는 1라운드 (참여형 토론)."""
    persona_list = _p.for_mode(mode)
    user_turn = {"name": "폐하", "text": question}
    for turn in run_round(persona_list, context, "reply", llm, prior_turns=[user_turn]):
        yield turn
