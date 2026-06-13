# corvin_jarvis/prediction/backtest.py
"""Phase 2 백테스트 게이트 — walk-forward 검증 + 통과 판정.

설계 원칙(spec): 각 방법론을 과거로 검증해 baseline 초과분만 다이제스트에 합류.
- 방향계(logistic/ensemble): directional hit-rate > baseline + margin
- 분포/밴드계(montecarlo/band): 실현값이 예측구간에 든 coverage ≈ target ±tol
결과는 state/phase2_backtest.json. 오케스트레이터가 읽어 통과 모델만 출력.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from corvin_jarvis.prediction import backfill, m_band, m_logistic, m_montecarlo

GATE_PATH = Path(__file__).resolve().parent.parent / "state" / "phase2_backtest.json"


# ── 순수 스코어러 ──────────────────────────────────────

def directional_score(predictions: list[tuple[float, int]],
                      margin: float = 0.03) -> dict[str, Any]:
    """predictions = [(prob_up, realized_up)]. baseline = 다수클래스 적중률."""
    n = len(predictions)
    if n == 0:
        return {"hit_rate": 0.0, "baseline": 0.0, "n": 0, "passed": False}
    hits = sum(1 for p, y in predictions if (1 if p >= 0.5 else 0) == y)
    hit_rate = hits / n
    up = sum(y for _, y in predictions) / n
    baseline = max(up, 1 - up)
    return {"hit_rate": hit_rate, "baseline": baseline, "n": n,
            "passed": hit_rate > baseline + margin}


def coverage_score(in_band_flags: list[bool], target: float = 0.8,
                   tol: float = 0.1) -> dict[str, Any]:
    """예측구간 적중 비율이 target ±tol 안이면 통과."""
    n = len(in_band_flags)
    if n == 0:
        return {"coverage": 0.0, "target": target, "n": 0, "passed": False}
    cov = sum(1 for f in in_band_flags if f) / n
    return {"coverage": cov, "target": target, "n": n,
            "passed": abs(cov - target) <= tol}


# ── walk-forward 러너 ──────────────────────────────────

def _base_closes(closes_by_feature: dict, base: str) -> list[float]:
    return [d["close"] for d in closes_by_feature.get(base, [])]


def walk_forward_logistic(closes_by_feature: dict[str, list[dict]], *,
                          features: list[str], horizon: int = 5,
                          min_train: int = 250, step: int = 20) -> dict[str, Any]:
    base = _base_closes(closes_by_feature, features[0])
    series_len = min((len(closes_by_feature.get(f, [])) for f in features),
                     default=0)
    preds: list[tuple[float, int]] = []
    for t in range(min_train, series_len - horizon, step):
        sliced = {f: closes_by_feature[f][:t + 1] for f in features}
        r = m_logistic.predict_market(sliced, features=features,
                                      min_days=min_train, horizon=horizon)
        if not r.data_ok:
            continue
        if base[t] <= 0:
            continue
        fwd = base[t + horizon] / base[t] - 1.0
        preds.append((r.evidence["prob_up"], 1 if fwd > 0 else 0))
    return directional_score(preds)


def walk_forward_coverage(closes: list[float], *, kind: str, horizon: int = 21,
                          min_train: int = 150, step: int = 20) -> dict[str, Any]:
    """band/montecarlo의 [p_low,p_high] 구간이 실현 H일 수익률을 포함하는 비율."""
    flags: list[bool] = []
    rng = np.random.default_rng(12345)
    for t in range(min_train, len(closes) - horizon, step):
        hist = closes[:t + 1]
        price = closes[t]
        if price <= 0:
            continue
        realized = closes[t + horizon] / price - 1.0
        if kind == "band":
            r = m_band.run_symbol("bt", hist, horizon=horizon, min_days=min_train)
            if not r.data_ok:
                continue
            lo = r.evidence["low"] / price - 1.0
            hi = r.evidence["high"] / price - 1.0
        else:  # montecarlo p5~p95
            r = m_montecarlo.run_symbol("bt", hist, horizon=horizon,
                                        min_days=min_train, n_paths=800, rng=rng)
            if not r.data_ok:
                continue
            lo, hi = r.evidence["p5"] / 100.0, r.evidence["p95"] / 100.0
        flags.append(lo <= realized <= hi)
    # band low_q/high_q=10/90 → 목표 coverage 0.8, montecarlo p5~p95 → 0.9
    target = 0.8 if kind == "band" else 0.9
    return coverage_score(flags, target=target, tol=0.12)


def run_all(db_path: Path, *, features: list[str], holdings: list[str]
            ) -> dict[str, Any]:
    """실 daily_history로 Phase 2 방법론 백테스트 → 게이트 dict."""
    feat_closes = {f: backfill.read_daily(db_path, f, 2000) for f in features}
    gate: dict[str, Any] = {}
    gate["logistic"] = walk_forward_logistic(feat_closes, features=features)
    base = features[0]
    base_closes = [d["close"] for d in feat_closes.get(base, [])]
    gate["band"] = walk_forward_coverage(base_closes, kind="band")
    gate["montecarlo"] = walk_forward_coverage(base_closes, kind="montecarlo")
    return gate


def save_gate(gate: dict[str, Any], path: Path = GATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")


def load_gate(path: Path = GATE_PATH) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def passed(gate: dict[str, Any], system: str) -> bool:
    return bool(gate.get(system, {}).get("passed"))
