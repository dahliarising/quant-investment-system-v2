"""Phase 2 통합 — 촉발원인 오케스트레이터.

jarvis 스냅샷(latest.json) → 원인 후보·랭킹·메시지 → state/cause.json 저장.
새 네트워크 호출 없음(이미 수집한 스냅샷 재활용). jarvis 펄스에서 호출.
"""
from __future__ import annotations

import json
from pathlib import Path

from corvin_jarvis import cause_attribution as ca


def run(snapshot: dict, state_path: Path) -> dict:
    """스냅샷 → 원인 메시지 + 후보. state 저장."""
    candidates = ca.candidates_from_snapshot(snapshot)
    ranked = ca.rank_causes(candidates)
    message = ca.attribution_message(ranked, top_n=3)
    out = {"message": message, "causes": ranked}
    state_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
