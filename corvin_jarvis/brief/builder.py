"""설득 브리핑 빌더 — 컴포넌트 조합. 데이터 로딩은 호출자 책임(주입)."""
from __future__ import annotations

from typing import Any

from corvin_jarvis.brief.evidence import build_evidence
from corvin_jarvis.brief.framing import build_framing
from corvin_jarvis.brief.position_overlay import build_position_lines
from corvin_jarvis.brief.psychology import build_psych_guard
from corvin_jarvis.brief.types import Brief, EvidenceStack


def build_brief(*, positions: list[dict[str, Any]], actions: list[dict[str, Any]],
                calibration: dict[str, Any], held: set[str],
                market_state: str, fresh_label: str, as_of: str) -> Brief:
    lines = build_position_lines(positions, fresh_label=fresh_label)

    evidence: dict[str, EvidenceStack] = {}
    for a in actions:
        sym = str(a.get("symbol", ""))
        if not sym:
            continue
        engine = (a.get("sources") or ["?"])[0]
        evidence[sym] = build_evidence(calibration, engine=engine,
                                       kind=str(a.get("action", "")))

    framing = build_framing(actions)

    # symbol 없는 액션은 제외 — str(None)='None'이 held 미포함이라 거짓 FOMO 트리거 방지
    has_new_buy = any(a.get("action") == "매수후보" and a.get("symbol")
                      and str(a.get("symbol")) not in held for a in actions)
    psych = build_psych_guard(positions, has_new_buy_candidate=has_new_buy)

    headline = f"🦅 Corvin 실행 브리핑 — {as_of.split(' ')[0][5:]}"
    return Brief(headline=headline, positions=lines, evidence=evidence,
                 framing=framing, psych=psych, as_of=as_of,
                 market_state=market_state)
