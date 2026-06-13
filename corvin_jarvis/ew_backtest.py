"""Phase 1 — EW 임계 백테스트.

과거 시계열에 프로덕션과 *동일한* 분류 코어를 흘려보내, 선행 신호가 실제로
드로다운을 앞섰는지(precision/recall/lead-time)와 거짓경보율을 측정.
순수 분석 코어(replay·label_drawdowns·evaluate·sweep)는 네트워크 없음.

(폐하 2026-06-06: EW 임계값이 백테스트 0인 초기 추정치 → 실제 검증.)
"""
from __future__ import annotations

import copy

from corvin_jarvis import early_warning as ew
from corvin_jarvis.ew_runner import _classify_all   # 프로덕션 분류 재사용(단일 진실원천)


def replay(history: list[dict], cfg: dict) -> list[dict]:
    """일별 readings → 프로덕션 분류·게이지·전일대비 전환 재현.

    history: [{date, readings, spx}] (날짜 오름차순). 순수함수만 호출.
    """
    out = []
    prev_states: dict = {}
    for rec in history:
        states = _classify_all(rec["readings"], cfg)
        gauge, reds = ew.composite_gauge(states)
        transitions = ew.detect_transitions(prev_states, states)
        out.append({
            "date": rec["date"],
            "states": states,
            "gauge": gauge,
            "reds": reds,
            "transitions": transitions,
        })
        prev_states = states
    return out


def label_drawdowns(history: list[dict], horizon: int, thresh_pct: float) -> dict:
    """미래 horizon일 내 SPX 드로다운 ≥ |thresh| 인 날을 '이벤트'로 라벨.

    lookahead은 *라벨링 전용* (신호 생성엔 미사용 — bias 가드).
    반환: {date: {drawdown_pct, lead_days}} (이벤트 날만).
    """
    events: dict[str, dict] = {}
    for i, rec in enumerate(history):
        base = rec["spx"]
        window = history[i + 1: i + 1 + horizon]
        if not window:
            continue
        troughs = [(j + 1, w["spx"]) for j, w in enumerate(window)]
        off, low = min(troughs, key=lambda t: t[1])
        drawdown = (low / base - 1) * 100
        if drawdown <= thresh_pct:
            events[rec["date"]] = {"drawdown_pct": drawdown, "lead_days": off}
    return events


def evaluate(signal_dates: set, events: dict) -> dict:
    """선행 신호 vs 실제 드로다운 이벤트 → precision/recall/lead-time/거짓경보.

    signal_dates·events 둘 다 '미래 조건이 성립하는 날' → 교집합이 적중.
    """
    sig = set(signal_dates)
    ev = set(events)
    tp_dates = sig & ev
    tp = len(tp_dates)
    precision = tp / len(sig) if sig else 0.0
    recall = tp / len(ev) if ev else 0.0
    leads = [events[d]["lead_days"] for d in tp_dates]
    lead_avg = sum(leads) / len(leads) if leads else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": (len(sig) - tp) / len(sig) if sig else 0.0,
        "lead_time_avg": lead_avg,
        "n_signals": len(sig),
        "n_events": len(ev),
    }


def build_history(series: dict[str, list[float]], dates: list[str],
                  warmup: int = 200) -> list[dict]:
    """정렬된 일별 시계열 → 일별 readings 레코드(순수 계산).

    series: SOXX·SPY·SPX·VIX·VIX3M·HY·CURVE(아래 일별 종가, dates와 동일 길이).
    breadth는 유니버스 전체 시계열이 필요해 백테스트 1차에선 제외(semis·vix·hy·curve 4지표).
    각 readings는 t시점 정보만 사용(미래 누설 없음).
    """
    soxx, spy = series["SOXX"], series["SPY"]
    vix, vix3m = series["VIX"], series["VIX3M"]
    hy, curve, spx = series["HY"], series["CURVE"], series["SPX"]
    ratio = [s / p for s, p in zip(soxx, spy)]

    out = []
    for i in range(warmup, len(dates)):
        ma50 = sum(ratio[i - 49:i + 1]) / 50
        slope_5d = ratio[i] - ratio[i - 5]
        lookback = spy[max(0, i - 251):i + 1]
        spx_dist = (spy[i] / max(lookback) - 1) * 100
        readings = {
            "semis": {"ratio": ratio[i], "ratio_ma50": ma50, "slope_5d": slope_5d,
                      "spx_dist_from_high_pct": spx_dist},
            "vix_term": {"ratio": vix[i] / vix3m[i]},
            "hy": {"value": hy[i], "chg_5d": hy[i] - hy[i - 5]},
            "curve": {"value": curve[i], "chg_5d": curve[i] - curve[i - 5]},
        }
        out.append({"date": dates[i], "readings": readings, "spx": spx[i]})
    return out


def signal_dates(replay_out: list[dict], predicate) -> set:
    """replay 결과에서 predicate가 참인 날짜 집합."""
    return {r["date"] for r in replay_out if predicate(r)}


# 신호 정의: 게이지 경보 + 지표별 RED 전환
PREDICATES = {
    "gauge_warn": lambda r: r["gauge"] in (ew.REDUCE, ew.SELL),
    "semis_red": lambda r: r["states"].get("semis") == ew.RED,
    "vix_term_red": lambda r: r["states"].get("vix_term") == ew.RED,
    "breadth_red": lambda r: r["states"].get("breadth") == ew.RED,
    "hy_red": lambda r: r["states"].get("hy") == ew.RED,
    "curve_red": lambda r: r["states"].get("curve") == ew.RED,
}


def run_backtest(history: list[dict], cfg: dict, horizon: int,
                 thresh_pct: float, predicates: dict | None = None) -> dict:
    """replay → 드로다운 라벨 → 신호별 평가. 신호종류별 metrics dict 반환."""
    preds = predicates or PREDICATES
    rep = replay(history, cfg)
    events = label_drawdowns(history, horizon, thresh_pct)
    return {name: evaluate(signal_dates(rep, pred), events)
            for name, pred in preds.items()}


def sweep_param(history: list[dict], base_cfg: dict, section: str, key: str,
                values: list, signal_name: str, horizon: int,
                thresh_pct: float) -> list[dict]:
    """한 임계 파라미터를 values로 스윕 → 해당 신호의 F1 기준 랭킹.

    base_cfg는 변경하지 않는다(deepcopy). 더 나은 임계를 찾거나, 어떤 값으로도
    엣지가 안 나오면(예: semis) 폐기 근거.
    """
    out = []
    for v in values:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault(section, {})[key] = v
        m = run_backtest(history, cfg, horizon, thresh_pct).get(signal_name, {})
        out.append({"value": v, "f1": m.get("f1", 0.0),
                    "precision": m.get("precision", 0.0),
                    "recall": m.get("recall", 0.0),
                    "lead_time_avg": m.get("lead_time_avg", 0.0)})
    return sorted(out, key=lambda r: r["f1"], reverse=True)
