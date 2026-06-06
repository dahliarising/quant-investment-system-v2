"""선행 경보 오케스트레이션 — readings(dict) → 분류·전환·게이지·하드스톱 → alert dict + state 저장.

라이브 readings 빌더(build_live_readings)는 ew_providers를 묶는 IO 경계.
순수 로직은 early_warning에 위임.
"""
from __future__ import annotations

import json
from pathlib import Path

from corvin_jarvis import early_warning as ew
from corvin_jarvis import ew_providers as ewp

_FRED = {"hy": "BAMLH0A0HYM2", "curve": "T10Y2Y"}


def _classify_all(readings: dict, cfg: dict) -> dict:
    states = {}
    if r := readings.get("semis"):
        states["semis"] = ew.classify_semis(r["ratio"], r["ratio_ma50"], r["slope_5d"],
                                             r["spx_dist_from_high_pct"], cfg)
    if r := readings.get("vix_term"):
        states["vix_term"] = ew.classify_vix_term(r["ratio"], cfg)
    if r := readings.get("breadth"):
        states["breadth"] = ew.classify_breadth(r["pct_above_ma200"], cfg)
    if r := readings.get("hy"):
        states["hy"] = ew.classify_hy(r["value"], r["chg_5d"], cfg)
    if r := readings.get("curve"):
        states["curve"] = ew.classify_curve(r["value"], r["chg_5d"], cfg)
    return states


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run(readings: dict, positions: list[dict], cfg: dict, state_path: Path) -> dict:
    states = _classify_all(readings, cfg)
    saved = _load_state(state_path)
    prev = saved.get("states", {})
    prev_gauge = saved.get("gauge")

    transitions = ew.detect_transitions(prev, states)
    gauge, reds = ew.composite_gauge(states)
    held = bool(positions)

    alerts = []
    for t in transitions:                    # 개별 지표 악화 전환
        sev = "high" if t["to"] == ew.RED else "medium"
        alerts.append({"key": t["key"], "severity": sev, "category": "early_warning",
                       "message": ew.indicator_message(t["key"], t["to"])})

    if gauge in (ew.REDUCE, ew.SELL) and gauge != prev_gauge:   # 게이지 악화 → 즉시 푸시
        alerts.append({"key": "gauge",
                       "severity": "critical" if gauge == ew.SELL else "high",
                       "category": "early_warning",
                       "message": f"게이지 {gauge}: {ew.action_label(gauge, held)} (적색 {reds}/5)"})

    for h in ew.hard_stop(positions, cfg["hard_stop_pct"]):     # -8% 하드스톱
        alerts.append({"key": f"hardstop_{h['sym']}", "severity": "critical",
                       "category": "early_warning", "message": h["message"]})

    state_path.write_text(json.dumps({"states": states, "gauge": gauge}, ensure_ascii=False),
                          encoding="utf-8")
    return {"alerts": alerts, "gauge": gauge, "states": states}


def build_live_readings(universe: list[str], spx_high: float | None) -> dict:
    """ew_providers 라이브 fetcher를 묶어 readings dict 생성. 누락 지표는 자동 스킵."""
    r = {}
    if semis := ewp.semis_reading(ewp.live_closes_fetcher, lambda: spx_high):
        r["semis"] = semis
    if vt := ewp.vix_term_reading(ewp.live_vix_fetcher):
        r["vix_term"] = vt
    if br := ewp.breadth_reading(universe, ewp.live_closes_fetcher):
        r["breadth"] = br
    if hy := ewp.fred_reading(ewp.live_fred_fetcher, _FRED["hy"]):
        r["hy"] = hy
    if cv := ewp.fred_reading(ewp.live_fred_fetcher, _FRED["curve"]):
        r["curve"] = cv
    return r
