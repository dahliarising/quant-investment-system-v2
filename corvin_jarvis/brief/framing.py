"""다각 프레이밍 — bull/bear/base + counterfactual (E1 템플릿 기반)."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.types import Framing

_BEAR = {"매도검토", "비중축소"}
_BULL = {"매수후보", "유지", "추가존"}


def _rep_rationale(actions: list[dict[str, Any]], kinds: set[str]) -> str:
    for a in sorted(actions, key=lambda x: x.get("urgency", 0), reverse=True):
        if a.get("action") in kinds and a.get("rationale"):
            return str(a["rationale"])
    return ""


def build_framing(actions: list[dict[str, Any]]) -> Framing:
    bear_n = sum(1 for a in actions if a.get("action") in _BEAR)
    bull_n = sum(1 for a in actions if a.get("action") in _BULL)
    bull_r = _rep_rationale(actions, _BULL)
    bear_r = _rep_rationale(actions, _BEAR)

    bull = f"강세축 {bull_n}건" + (f" — {bull_r}" if bull_r else "")
    bear = f"방어축 {bear_n}건" + (f" — {bear_r}" if bear_r else "")
    base = f"base: 방어 {bear_n} · 매수후보/유지 {bull_n}"

    if actions:
        top = max(actions, key=lambda x: x.get("urgency", 0))
        cf = (f"지금 {top.get('symbol')} {top.get('action')} 미실행 시: "
              f"urgency {top.get('urgency', 0)} 신호 방치 — {top.get('rationale', '')}")
    else:
        cf = "활성 액션 없음 — 관망이 base."
    return Framing(bull=bull, bear=bear, base=base, counterfactual=cf)
