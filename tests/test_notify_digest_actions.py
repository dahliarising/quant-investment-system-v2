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


def test_actions_block_caps_actionable_rows(tmp_path):
    """액션성(관찰 등)은 6건 캡 + 나머지 요약 (매수후보는 별도 한 줄)."""
    actions = [{"symbol": f"S{i}", "action": "관찰", "urgency": 40,
                "rationale": "w", "sources": [], "conflict": False} for i in range(12)]
    p = _write_state(tmp_path, datetime.now(KST), actions)
    block = notify._final_actions_block(state_path=p)
    assert block.count("👀") == 6
    assert "외 액션 6건" in block


def test_actions_block_stale_with_actionable_rows_still_empty(tmp_path):
    p = _write_state(tmp_path, datetime.now(KST) - timedelta(hours=25), [
        {"symbol": "X", "action": "매도검토", "urgency": 95, "rationale": "r",
         "sources": [], "conflict": False},
    ])
    assert notify._final_actions_block(state_path=p) == ""


# ── 매수후보 요약 + 예측 블록 (2026-06-11) ──────────────

def _write_state_full(tmp_path, ts, actions, predictive=None):
    p = tmp_path / "final_actions.json"
    obj = {"ts": ts.isoformat(timespec="seconds"), "actions": actions}
    if predictive is not None:
        obj["predictive"] = predictive
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def test_buy_candidates_collapsed_to_summary(tmp_path):
    """매수후보 다수 → 한 줄 요약 (종목 과다 해소). 액션성은 개별 노출."""
    actions = [{"symbol": "012450", "action": "매도검토", "urgency": 95,
                "rationale": "손절", "sources": ["x"], "conflict": False}]
    actions += [{"symbol": f"00{i}", "action": "매수후보", "urgency": 50,
                 "rationale": "Z3 딥밸류", "sources": ["playbook"], "conflict": False}
                for i in range(10)]
    block = notify._final_actions_block(state_path=_write_state_full(tmp_path, datetime.now(KST), actions))
    assert "012450" in block and "매도검토" in block      # 액션성 개별
    assert "매수후보 10종목" in block                      # 요약 한 줄
    # 10개가 개별 줄로 안 나와야 (요약됨)
    assert block.count("Z3 딥밸류") <= 1


def test_predictive_block_formats_signals(tmp_path):
    preds = [
        {"kind": "EVENT", "symbol": "", "message": "FOMC 금리결정 D-6 — 변동성", "horizon_days": 6, "urgency": 50},
        {"kind": "RS_WEAK", "symbol": "NVDA", "message": "20일 S&P500 대비 -7.5%p", "horizon_days": None, "urgency": 44},
    ]
    p = _write_state_full(tmp_path, datetime.now(KST), [], predictive=preds)
    block = notify._predictive_block(state_path=p)
    assert "예측" in block
    assert "FOMC" in block and "📅" in block
    assert "NVDA" in block and "📉" in block


def test_predictive_block_empty_when_none(tmp_path):
    p = _write_state_full(tmp_path, datetime.now(KST), [], predictive=[])
    assert notify._predictive_block(state_path=p) == ""
