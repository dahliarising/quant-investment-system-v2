"""LangGraph StateGraph — Bull→Bear(parallel)→Researcher→Executor."""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from .state import TradingState
from .nodes import bull_node, bear_node, researcher_node, executor_node

log = logging.getLogger("corvin.agents.graph")


def _build_graph() -> Any:
    g = StateGraph(TradingState)
    g.add_node("bull", bull_node)
    g.add_node("bear", bear_node)
    g.add_node("researcher", researcher_node)
    g.add_node("executor", executor_node)

    g.set_entry_point("bull")
    g.add_edge("bull", "bear")
    g.add_edge("bear", "researcher")
    g.add_edge("researcher", "executor")
    g.add_edge("executor", END)

    return g.compile()


_graph = None


def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_agent_analysis(
    symbol: str,
    latest: dict[str, Any],
    alerts: list[dict[str, Any]],
) -> dict[str, Any]:
    """단일 종목에 대한 4-에이전트 분석 실행. 결과 dict 반환."""
    init_state: TradingState = {
        "symbol": symbol,
        "latest": latest,
        "alerts": alerts,
        "bull_thesis": "",
        "bear_thesis": "",
        "research_summary": "",
        "confidence": 0.5,
        "action": "hold",
        "rationale": "",
    }
    result = _get_graph().invoke(init_state)
    log.info("[agents] %s → action=%s confidence=%.2f", symbol, result["action"], result["confidence"])
    return result
