"""Bull / Bear / Researcher / Executor 에이전트 노드."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from .state import TradingState
from .tools import (
    get_bull_context,
    get_bear_context,
    get_history,
    get_relative_strength,
    get_downside_probability,
    get_regime,
)

log = logging.getLogger("corvin.agents")

_client: anthropic.Anthropic | None = None


def _claude() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _call(system: str, user: str, max_tokens: int = 512) -> str:
    msg = _claude().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


# ── Bull Analyst ──────────────────────────────────────────────────────────────

def bull_node(state: TradingState) -> TradingState:
    symbol = state["symbol"]
    ctx = get_bull_context(symbol)
    hist = get_history(symbol)
    rel = get_relative_strength(symbol)

    signals_text = "\n".join(f"- {s}" for s in ctx.get("signals", [])) or "- 유의미한 강세 신호 없음"
    prompt = f"""종목: {symbol}
7일 수익률: {hist.get('pct_change', 0):+.2f}%
vs SP500: {rel.get('verdict', 'N/A')} ({rel.get('relative_pp', 0):+.2f}pp)
강세 신호:
{signals_text}

위 데이터를 바탕으로 매수/보유 논거를 2-3문장으로 작성하라. 한국어."""

    thesis = _call(
        "당신은 강세론(bull) 주식 애널리스트입니다. 데이터를 근거로 매수 논거를 제시하십시오.",
        prompt,
    )
    log.info("[bull] %s → %s...", symbol, thesis[:60])
    return {**state, "bull_thesis": thesis}


# ── Bear Analyst ──────────────────────────────────────────────────────────────

def bear_node(state: TradingState) -> TradingState:
    symbol = state["symbol"]
    ctx = get_bear_context(symbol)
    hist = get_history(symbol)
    downside = get_downside_probability(symbol, threshold_pct=-0.05)

    signals_text = "\n".join(f"- {s}" for s in ctx.get("signals", [])) or "- 유의미한 약세 신호 없음"
    prompt = f"""종목: {symbol}
7일 수익률: {hist.get('pct_change', 0):+.2f}%
-5% 이하 확률(30일): {downside:.1%}
약세 신호:
{signals_text}

위 데이터를 바탕으로 매도/관망 논거를 2-3문장으로 작성하라. 한국어."""

    thesis = _call(
        "당신은 약세론(bear) 주식 애널리스트입니다. 데이터를 근거로 리스크를 부각하십시오.",
        prompt,
    )
    log.info("[bear] %s → %s...", symbol, thesis[:60])
    return {**state, "bear_thesis": thesis}


# ── Researcher ────────────────────────────────────────────────────────────────

def researcher_node(state: TradingState) -> TradingState:
    symbol = state["symbol"]
    regime = get_regime(state["latest"])
    alerts_text = "\n".join(
        f"- [{a.get('severity','?')}] {a.get('message','')}"
        for a in state["alerts"]
        if a.get("symbol") == symbol or a.get("scope") == "global"
    ) or "- 해당 없음"

    prompt = f"""종목: {symbol}
레짐: {regime.get('label','neutral')} (score={regime.get('score',0.5):.2f})
관련 알림:
{alerts_text}

강세 논거: {state['bull_thesis']}
약세 논거: {state['bear_thesis']}

양측 논거와 매크로 환경을 종합하여 확신도(0.0~1.0)와 핵심 판단 근거를 제시하라.
JSON으로 반환: {{"confidence": 0.0~1.0, "summary": "2문장 요약"}}"""

    raw = _call(
        "당신은 리서치 총괄입니다. Bull/Bear 논거와 매크로 시그널을 종합하여 판단하십시오.",
        prompt,
        max_tokens=256,
    )
    try:
        parsed = json.loads(raw)
        confidence = float(parsed.get("confidence", 0.5))
        summary = parsed.get("summary", raw)
    except Exception:
        confidence = 0.5
        summary = raw

    return {**state, "confidence": confidence, "research_summary": summary}


# ── Executor ──────────────────────────────────────────────────────────────────

def executor_node(state: TradingState) -> TradingState:
    confidence = state["confidence"]
    bull = state["bull_thesis"]
    bear = state["bear_thesis"]
    symbol = state["symbol"]

    prompt = f"""종목: {symbol}, 확신도: {confidence:.2f}
강세 논거: {bull}
약세 논거: {bear}
리서치 요약: {state['research_summary']}

최종 행동 판정:
- confidence > 0.65 → buy
- confidence < 0.35 → sell
- 그 외 → hold

JSON: {{"action": "buy|sell|hold", "rationale": "1문장"}}"""

    raw = _call(
        "당신은 포트폴리오 집행 에이전트입니다. 리스크를 고려하여 최종 행동을 결정하십시오.",
        prompt,
        max_tokens=128,
    )
    try:
        parsed = json.loads(raw)
        action = parsed.get("action", "hold")
        rationale = parsed.get("rationale", raw)
    except Exception:
        action = "hold"
        rationale = raw

    if action not in ("buy", "sell", "hold"):
        action = "hold"

    return {**state, "action": action, "rationale": rationale}
