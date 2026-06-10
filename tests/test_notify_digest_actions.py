"""notify digest — 중재 최종 액션 블록 (state/final_actions.json 기반)."""
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from corvin_jarvis import notify

KST = ZoneInfo("Asia/Seoul")


def _write_state(tmp_path, ts, actions):
    p = tmp_path / "final_actions.json"
    p.write_text(json.dumps({"ts": ts.isoformat(timespec="seconds"),
                             "actions": actions}, ensure_ascii=False), encoding="utf-8")
    return p


def test_actions_block_renders_with_icons(tmp_path):
    p = _write_state(tmp_path, datetime.now(KST), [
        {"symbol": "012450", "action": "매도검토", "urgency": 95,
         "rationale": "손절선 이탈 — 매수 신호(playbook) 상충, 안전 우선",
         "sources": ["playbook", "signal_engine"], "conflict": True},
        {"symbol": "GOOGL", "action": "관찰", "urgency": 40,
         "rationale": "RS 약세", "sources": ["predictive"], "conflict": False},
    ])
    block = notify._final_actions_block(state_path=p)
    assert "🛑" in block and "012450" in block and "매도검토" in block
    assert "👀" in block and "GOOGL" in block
    assert "⚔️" in block  # conflict 표시


def test_actions_block_empty_when_stale(tmp_path):
    p = _write_state(tmp_path, datetime.now(KST) - timedelta(hours=25), [
        {"symbol": "X", "action": "홀딩", "urgency": 20, "rationale": "", "sources": [], "conflict": False},
    ])
    assert notify._final_actions_block(state_path=p) == ""


def test_actions_block_empty_when_missing(tmp_path):
    assert notify._final_actions_block(state_path=tmp_path / "nope.json") == ""


def test_actions_block_skips_hold_only(tmp_path):
    """홀딩만 있으면 노이즈 — 블록 생략 (alert noise aversion)."""
    p = _write_state(tmp_path, datetime.now(KST), [
        {"symbol": "MSFT", "action": "홀딩", "urgency": 20, "rationale": "정상",
         "sources": ["signal_engine"], "conflict": False},
    ])
    assert notify._final_actions_block(state_path=p) == ""


def test_actions_block_caps_rows_at_8(tmp_path):
    actions = [{"symbol": f"S{i}", "action": "관찰", "urgency": 40,
                "rationale": "w", "sources": [], "conflict": False} for i in range(12)]
    p = _write_state(tmp_path, datetime.now(KST), actions)
    block = notify._final_actions_block(state_path=p)
    assert block.count("👀") == 8
    assert "외 4건" in block


def test_actions_block_stale_with_actionable_rows_still_empty(tmp_path):
    p = _write_state(tmp_path, datetime.now(KST) - timedelta(hours=25), [
        {"symbol": "X", "action": "매도검토", "urgency": 95, "rationale": "r",
         "sources": [], "conflict": False},
    ])
    assert notify._final_actions_block(state_path=p) == ""
