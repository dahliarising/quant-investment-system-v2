"""Phase 1 — EW 임계 백테스트 테스트.

순수 분석 코어(replay·label_drawdowns·evaluate·sweep)는 주입 데이터로 전부
네트워크 없이 검증. replay는 프로덕션과 동일한 분류함수 재사용.
"""
from corvin_jarvis import ew_backtest as bt
from corvin_jarvis import early_warning as ew

CFG = {
    "vix_term": {"green_max": 0.9, "amber_max": 1.0},
    "breadth": {"green_min": 60.0, "amber_min": 40.0},
    "hy": {"green_max": 3.5, "amber_max": 5.0, "rise_amber_5d": 0.3, "rise_red_5d": 0.5},
    "curve": {"green_min": 0.5, "amber_min": 0.0, "fast_move_5d": 0.15},
    "semis": {"divergence_high_dist_pct": 3.0},
}


def _green_readings():
    return {
        "semis": {"ratio": 1.10, "ratio_ma50": 1.05, "slope_5d": 0.02, "spx_dist_from_high_pct": -1.0},
        "vix_term": {"ratio": 0.85},
        "breadth": {"pct_above_ma200": 70},
        "hy": {"value": 2.0, "chg_5d": 0.0},
        "curve": {"value": 0.8, "chg_5d": 0.0},
    }


def _semis_red_readings():
    r = _green_readings()
    r["semis"] = {"ratio": 1.0, "ratio_ma50": 1.05, "slope_5d": -0.02, "spx_dist_from_high_pct": -1.0}
    return r


def test_replay_reproduces_gauge_and_transitions():
    history = [
        {"date": "2026-06-01", "readings": _green_readings(), "spx": 100.0},
        {"date": "2026-06-02", "readings": _semis_red_readings(), "spx": 99.0},
    ]
    out = bt.replay(history, CFG)
    assert len(out) == 2
    assert out[0]["gauge"] == ew.BUY                 # day1 all green
    assert out[1]["gauge"] == ew.REDUCE              # semis RED → REDUCE
    keys = [t["key"] for t in out[1]["transitions"]]
    assert "semis" in keys                            # green→red transition detected
    assert out[1]["states"]["semis"] == ew.RED


# ── label_drawdowns: 미래 드로다운 이벤트 라벨 (lookahead은 라벨 전용) ──
def test_label_drawdowns_marks_days_preceding_drop():
    history = [
        {"date": "d0", "spx": 100.0}, {"date": "d1", "spx": 99.0},
        {"date": "d2", "spx": 95.0},  {"date": "d3", "spx": 101.0},
    ]
    events = bt.label_drawdowns(history, horizon=2, thresh_pct=-3.0)
    assert set(events) == {"d0", "d1"}                # both followed by ≥3% drop within 2d
    assert events["d0"]["lead_days"] == 2             # trough at d2
    assert events["d1"]["lead_days"] == 1
    assert events["d0"]["drawdown_pct"] < -3.0


# ── evaluate: precision / recall / lead-time / 거짓경보 ──────────────
def test_evaluate_precision_recall_leadtime():
    events = {"d0": {"lead_days": 2}, "d1": {"lead_days": 1}}   # 2 real events
    signal_dates = {"d0", "dX"}                                  # d0 hit, dX false alarm
    m = bt.evaluate(signal_dates, events)
    assert m["tp"] == 1
    assert m["precision"] == 0.5                                 # 1 of 2 signals real
    assert m["recall"] == 0.5                                    # caught 1 of 2 events
    assert m["false_alarm_rate"] == 0.5
    assert m["lead_time_avg"] == 2.0                            # only caught d0 (lead 2)


def test_evaluate_handles_empty_signals():
    m = bt.evaluate(set(), {"d0": {"lead_days": 1}})
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["tp"] == 0


# ── signal_dates + 통합 백테스트 ───────────────────────────────────
def test_signal_dates_extracts_by_predicate():
    rep = [
        {"date": "a", "gauge": ew.BUY, "states": {"semis": ew.GREEN}},
        {"date": "b", "gauge": ew.REDUCE, "states": {"semis": ew.RED}},
    ]
    assert bt.signal_dates(rep, lambda r: r["states"].get("semis") == ew.RED) == {"b"}


def test_run_backtest_catches_signal_before_drop():
    history = [
        {"date": "d0", "readings": _green_readings(), "spx": 100.0},
        {"date": "d1", "readings": _semis_red_readings(), "spx": 100.0},  # 신호
        {"date": "d2", "readings": _green_readings(), "spx": 96.0},        # 드롭 −4%
        {"date": "d3", "readings": _green_readings(), "spx": 97.0},
    ]
    res = bt.run_backtest(history, CFG, horizon=2, thresh_pct=-3.0)
    semis = res["semis_red"]
    assert semis["precision"] == 1.0          # 신호 d1만, 실제 드롭 앞섬
    assert semis["recall"] == 0.5             # 이벤트 2개 중 1개 적중(d1)
    assert semis["lead_time_avg"] == 1.0      # d1→트로프(d2) 1일 선행
    assert "gauge_warn" in res                # 게이지 경보도 평가됨


# ── build_history: 정렬 시계열 → 일별 readings (순수 계산) ──────────
def test_build_history_computes_daily_readings():
    n = 60
    series = {
        "SOXX": [90.0] * n, "SPY": [200.0] * n, "SPX": [200.0] * n,
        "VIX": [20.0] * n, "VIX3M": [21.0] * n,
        "HY": [3.0] * n, "CURVE": [0.5] * n,
    }
    dates = [f"2026-{i:03d}" for i in range(n)]
    hist = bt.build_history(series, dates, warmup=50)
    assert len(hist) == 10                      # i=50..59
    rec = hist[0]
    assert rec["date"] == dates[50]
    r = rec["readings"]
    assert abs(r["semis"]["ratio"] - 0.45) < 1e-9        # 90/200
    assert abs(r["semis"]["ratio_ma50"] - 0.45) < 1e-9
    assert abs(r["semis"]["slope_5d"]) < 1e-9            # constant
    assert abs(r["vix_term"]["ratio"] - (20.0 / 21.0)) < 1e-9
    assert r["hy"]["value"] == 3.0 and abs(r["hy"]["chg_5d"]) < 1e-9
    assert r["curve"]["value"] == 0.5
    assert rec["spx"] == 200.0
