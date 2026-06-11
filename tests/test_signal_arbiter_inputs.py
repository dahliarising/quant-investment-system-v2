# tests/test_signal_arbiter_inputs.py
"""arbiter_inputs — 엔진별 신호 → 공통 normalize dict."""
from corvin_jarvis.signals import arbiter_inputs as ai


def test_normalize_engine_signals():
    rows = ai.normalize_engine_signals([
        {"symbol": "012450", "kind": "STOP", "urgency": 95, "reason": "손절선 이탈"},
        {"symbol": "NVDA", "kind": "WATCH", "urgency": 70, "reason": "근접"},
        {"symbol": "META", "kind": "TRIM", "urgency": 55, "reason": "익절"},
        {"symbol": "MSFT", "kind": "HOLD", "urgency": 20, "reason": "정상"},
        {"symbol": "GOOGL", "kind": "UNKNOWN", "urgency": 0, "reason": "가격없음"},
    ])
    by = {r["symbol"]: r for r in rows}
    assert by["012450"]["intent"] == "defensive"
    assert by["NVDA"]["intent"] == "warn"
    assert by["META"]["intent"] == "trim"
    assert by["MSFT"]["intent"] == "hold"
    assert "GOOGL" not in by  # UNKNOWN 제외
    assert all(r["engine"] == "signal_engine" for r in rows)


def test_normalize_predictive_signals():
    rows = ai.normalize_predictive_signals([
        {"symbol": "012450", "kind": "VELOCITY", "urgency": 82, "message": "D-1 도달"},
        {"symbol": "BWXT", "kind": "RS_WEAK", "urgency": 45, "message": "상대약세"},
        {"symbol": "", "kind": "EVENT", "urgency": 80, "message": "FOMC D-1"},
    ])
    by = {r["symbol"]: r for r in rows}
    assert by["012450"]["intent"] == "warn"  # defensive 승격은 arbiter 규칙(urgency>=70)이 담당
    assert by["BWXT"]["intent"] == "warn"
    assert "" not in by  # 매크로 제외
    assert all(r["engine"] == "predictive" for r in rows)


def test_normalize_playbook_signals():
    rows = ai.normalize_playbook_signals([
        {"sym": "GOOGL", "zone": "딥밸류", "stance": "ENTER", "color": "green"},
        {"sym": "META", "zone": "고점", "stance": "HARVEST", "color": "amber"},
    ])
    by = {r["symbol"]: r for r in rows}
    assert by["GOOGL"]["intent"] == "buy" and by["GOOGL"]["kind"] == "BUY_NOW"
    assert by["META"]["intent"] == "trim" and by["META"]["kind"] == "TRIM_NOW"
    assert all(r["engine"] == "playbook" for r in rows)


def test_normalize_ledger_open_rows():
    rows = ai.normalize_ledger_open([
        {"engine": "leading", "symbol": "NVDA", "kind": "ensemble",
         "direction": "bull", "urgency": None, "confidence": 70.0,
         "evidence": {"message": "추세+RS"}},
        {"engine": "leading", "symbol": "012450", "kind": "ensemble",
         "direction": "bear", "urgency": None, "confidence": 40.0, "evidence": {}},
        {"engine": "jarvis", "symbol": "TSLA", "kind": "stop_loss",
         "direction": None, "urgency": 90, "confidence": None,
         "evidence": {"message": "STOP LOSS 도달"}},
        {"engine": "jarvis", "symbol": "", "kind": "kospi",
         "direction": None, "urgency": 90, "confidence": None, "evidence": {}},
        {"engine": "jarvis", "symbol": "NVDA", "kind": "rs_confirmed",
         "direction": None, "urgency": 55, "confidence": None, "evidence": {}},
        {"engine": "signal_engine", "symbol": "012450", "kind": "WATCH",
         "direction": None, "urgency": 70, "confidence": None, "evidence": {}},
    ])
    by = {(r["engine"], r["symbol"]): r for r in rows}
    assert by[("leading", "NVDA")]["intent"] == "buy"
    assert by[("leading", "012450")]["intent"] == "warn"
    assert by[("jarvis", "TSLA")]["intent"] == "defensive"
    assert ("jarvis", "") not in by                  # 매크로 제외
    assert ("jarvis", "NVDA") not in by              # 방향성 없는 정보성 제외
    assert ("signal_engine", "012450") not in by     # 라이브 엔진과 중복 — ledger의 engine/predictive/playbook행 제외


def test_build_final_actions_writes_state(tmp_path, monkeypatch):
    """collect→arbitrate→state 파일 쓰기 E2E (입력 전부 주입)."""
    out_path = tmp_path / "final_actions.json"
    result = ai.build_final_actions(
        engine_sigs=[{"symbol": "012450", "kind": "STOP", "urgency": 95, "reason": "이탈"}],
        pred_sigs=[],
        playbook_sigs=[{"sym": "012450", "zone": "딥밸류", "stance": "ENTER", "color": "green"}],
        ledger_open=[],
        calibration={},
        out_path=out_path,
    )
    assert result["actions"][0]["symbol"] == "012450"
    assert result["actions"][0]["action"] == "매도검토"
    assert result["actions"][0]["conflict"] is True
    import json
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["actions"] == result["actions"]
    assert "ts" in saved


def test_normalize_playbook_unknown_color_skipped():
    rows = ai.normalize_playbook_signals([
        {"sym": "XXX", "zone": "?", "stance": "ENTER"},            # color 없음
        {"sym": "YYY", "zone": "?", "stance": "ENTER", "color": "red"},  # 미지 색
    ])
    assert rows == []


def test_collect_live_survives_partial_failure(monkeypatch, tmp_path):
    """한 엔진 실패 → 빈 입력 강등, 중재는 계속 + state 파일 생성."""
    from corvin_jarvis.dashboard import snapshot as snap
    from corvin_jarvis.playbook import builder
    from corvin_jarvis.signals import calibration as cal_mod
    from corvin_jarvis.signals import ledger

    monkeypatch.setattr(ai, "STATE_PATH", tmp_path / "fa.json")
    monkeypatch.setattr(snap, "_load_portfolio", lambda: {"holdings": []})
    monkeypatch.setattr(snap, "_positions", lambda pf, fx=None: [])
    monkeypatch.setattr(snap, "_held_for_engine", lambda pf, pos: [])
    monkeypatch.setattr(snap, "_engine_signals",
                        lambda held: [{"symbol": "NVDA", "kind": "STOP", "urgency": 95, "reason": "이탈"}])
    def _boom(held):
        raise ConnectionError("pykrx down")
    monkeypatch.setattr(snap, "_predictive_signals", _boom)
    monkeypatch.setattr(builder, "load_holdings", lambda path=None: {})
    monkeypatch.setattr(snap, "_build_signals", lambda holdings: [])
    monkeypatch.setattr(ledger, "fetch_open", lambda db_path=None: [])
    monkeypatch.setattr(cal_mod, "compute", lambda db_path=None: {})

    res = ai.collect_live()
    assert res["actions"][0]["symbol"] == "NVDA"   # 살아남은 엔진으로 중재 완료
    assert (tmp_path / "fa.json").exists()


def test_normalize_ledger_excludes_dca_defensive():
    """dca 종목의 jarvis 방어(stop) 행은 제외 — 가격손절 면제 일관성."""
    rows = [
        {"engine": "jarvis", "symbol": "012450", "kind": "stop_loss",
         "urgency": 90, "confidence": 50, "evidence": {"message": "stop"}},
        {"engine": "jarvis", "symbol": "TSLA", "kind": "stop_loss",
         "urgency": 90, "confidence": 50, "evidence": {"message": "stop"}},
    ]
    out = ai.normalize_ledger_open(rows, dca_syms={"012450"})
    syms = {r["symbol"] for r in out}
    assert "012450" not in syms   # dca → 제외
    assert "TSLA" in syms          # trade → 유지
