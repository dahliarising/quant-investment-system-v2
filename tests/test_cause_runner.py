"""Phase 2 통합 — cause_runner 테스트.

jarvis 스냅샷 → 원인 후보·랭킹·메시지 → state 저장. 디스크 격리(tmp_path).
"""
import json

from corvin_jarvis import cause_runner


def test_run_builds_message_and_persists(tmp_path):
    snap = {
        "indices": {"vix": {"price": 21.5, "pct_change": 40.0}, "kospi": {"pct_change": -5.5}},
        "universe": [
            {"sector": "반도체", "pct_change": -5.0},
            {"sector": "반도체", "pct_change": -5.4},
            {"sector": "방어주", "pct_change": -0.1},
        ],
    }
    state = tmp_path / "cause.json"
    out = cause_runner.run(snap, state_path=state)
    assert "반도체" in out["message"]                 # 최강 원인
    assert "변동성" in out["message"]                  # VIX 급등
    assert state.exists()
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["message"] == out["message"]


def test_run_quiet_market_returns_unknown(tmp_path):
    snap = {"indices": {"vix": {"pct_change": 1.0}},
            "universe": [{"sector": "반도체", "pct_change": 0.2}]}
    out = cause_runner.run(snap, state_path=tmp_path / "c.json")
    assert "원인 불명" in out["message"]
