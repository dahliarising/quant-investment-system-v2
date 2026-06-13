import json
import pathlib
import tempfile

from corvin_jarvis import ew_runner

CFG = {
    "vix_term": {"green_max": 0.9, "amber_max": 1.0},
    "breadth": {"green_min": 60.0, "amber_min": 40.0},
    "hy": {"green_max": 3.5, "amber_max": 5.0, "rise_amber_5d": 0.3, "rise_red_5d": 0.5},
    "curve": {"green_min": 0.5, "amber_min": 0.0, "fast_move_5d": 0.15},
    "semis": {"divergence_high_dist_pct": 3.0},
    "hard_stop_pct": -8.0,
    "enabled": True,
}


def _readings_red_semis():
    return {
        "semis": {"ratio": 1.0, "ratio_ma50": 1.05, "slope_5d": -0.02, "spx_dist_from_high_pct": -1.0},
        "vix_term": {"ratio": 0.85},
        "breadth": {"pct_above_ma200": 70},
        "hy": {"value": 2.74, "chg_5d": 0.0},
        "curve": {"value": 0.4, "chg_5d": 0.0},
    }


# ── Task 9: runner ───────────────────────────────────────────
def test_run_emits_semis_transition_and_hardstop(tmp_path):
    state_file = tmp_path / "ew_state.json"
    out = ew_runner.run(
        readings=_readings_red_semis(),
        positions=[{"sym": "012450", "pnl_pct": -14.6}],
        cfg=CFG, state_path=state_file,
    )
    keys = [a["key"] for a in out["alerts"]]
    assert "semis" in keys                                         # 반도체 RED 전환 발화
    assert any("파세요 012450" in a["message"] for a in out["alerts"])  # 하드스톱
    assert out["gauge"] == "REDUCE"                                # semis red → REDUCE
    assert state_file.exists()


def test_second_run_no_duplicate(tmp_path):
    state_file = tmp_path / "ew_state.json"
    ew_runner.run(readings=_readings_red_semis(), positions=[], cfg=CFG, state_path=state_file)
    out2 = ew_runner.run(readings=_readings_red_semis(), positions=[], cfg=CFG, state_path=state_file)
    assert all(a["key"] != "semis" for a in out2["alerts"])        # 유지 → 재발화 X
    assert all(a["key"] != "gauge" for a in out2["alerts"])        # 게이지 동일 → 푸시 X


# ── Task 10: live readings builder ───────────────────────────
def test_build_readings_uses_providers(monkeypatch):
    from corvin_jarvis import ew_providers as ewp
    monkeypatch.setattr(ewp, "live_closes_fetcher",
                        lambda s, n: [90.0] * 200 if s == "SOXX" else [100.0] * 200)
    monkeypatch.setattr(ewp, "live_vix_fetcher",
                        lambda s: 21.5 if s == "^VIX" else 21.8)
    monkeypatch.setattr(ewp, "live_fred_fetcher",
                        lambda c, n: [2.74] * 8 if c == "BAMLH0A0HYM2" else [0.4] * 8)
    r = ew_runner.build_live_readings(universe=["AAA"], spx_high=205.0)
    assert "vix_term" in r and "hy" in r and "curve" in r


# ── Task 11: regression — Thursday semis leads crash by 1 day ─
def test_regression_thursday_semis_red_before_crash():
    readings = {
        "semis": {"ratio": 0.72, "ratio_ma50": 0.74, "slope_5d": -0.015, "spx_dist_from_high_pct": -1.2},
    }
    with tempfile.TemporaryDirectory() as d:
        out = ew_runner.run(readings=readings, positions=[], cfg=CFG,
                            state_path=pathlib.Path(d) / "s.json")
    msgs = " ".join(a["message"] for a in out["alerts"])
    assert "반도체 줄이세요" in msgs        # 지수 급락 1일 전 발화
