"""Phase D — SSE 스트림 (순수 로직).

debate 제너레이터의 각 턴을 fact-check → SSE 이벤트 문자열로 emit.
emit 콜백 DI → 소켓 없이 테스트. 서버는 emit에 wfile.write를 꽂기만.
"""
from __future__ import annotations

import json
from typing import Callable

from corvin_jarvis.live_debate import context as _ctx
from corvin_jarvis.live_debate import engine as _engine
from corvin_jarvis.live_debate import factcheck as _fc


def sse_event(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False, default=str) + "\n\n"


def run_stream(mode: str, context: dict, llm: Callable[[str], str],
               emit: Callable[[str], None], rounds: list[str] | None = None) -> None:
    """토론을 돌리며 턴마다 fact-check → SSE emit. 마지막에 done."""
    facts = _ctx.context_facts(context)
    for turn in _engine.debate(mode, context, llm, rounds):
        verdict = _fc.verify_turn(turn["text"], facts)
        annotated = _fc.annotate(turn, verdict)
        annotated["kind"] = "turn"
        emit(sse_event(annotated))
    emit(sse_event({"kind": "done"}))


def run_reply_stream(mode: str, context: dict, question: str,
                     llm: Callable[[str], str], emit: Callable[[str], None]) -> None:
    """폐하 질문 → 각 페르소나 응답 1라운드 → fact-check → SSE emit (참여형)."""
    facts = _ctx.context_facts(context)
    for turn in _engine.respond_to_user(mode, context, question, llm):
        annotated = _fc.annotate(turn, _fc.verify_turn(turn["text"], facts))
        annotated["kind"] = "turn"
        emit(sse_event(annotated))
    emit(sse_event({"kind": "done"}))
