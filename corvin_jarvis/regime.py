"""Corvin Jarvis — Regime Detector (Tier 2.2)

VIX + USD/KRW + (optional) yield spread / correlation → risk-on/neutral/risk-off/crisis 라벨.

매 pulse 후 호출. 라벨 전환 시 alert 자동 생성.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import statistics
from datetime import date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.regime")

LABELS = ("crisis", "risk_off", "neutral", "risk_on", "euphoria")


def label_from_signals(
    vix: float | None,
    vix_zscore: float | None,
    usd_krw_pct: float | None,
    correlation: float | None = None,
) -> dict[str, Any]:
    """순수 함수: 시그널 → label + score(-100~100) + drivers.

    Score 합산 방식 (각 -100~+100 기여):
    - VIX 절대값: 낮을수록 risk-on. 12 미만 +40 / 20 이상 -20 / 30 이상 -60 / 45 이상 -100
    - VIX zscore: 양수면 risk-off 가중치
    - USD/KRW pct: 1% 이상 약달러 vs 원약세 → risk-off
    - correlation: KOSPI-SPY 디커플링 (음수) → mixed risk
    """
    score = 0.0
    drivers: list[str] = []

    if vix is not None:
        if vix < 13:
            score += 30
            drivers.append(f"VIX {vix:.1f} (low, complacency)")
        elif vix < 18:
            score += 10
        elif vix < 25:
            drivers.append(f"VIX {vix:.1f} (elevated)")
        elif vix < 35:
            score -= 25
            drivers.append(f"VIX {vix:.1f} (high stress)")
        else:
            score -= 60
            drivers.append(f"VIX {vix:.1f} (crisis level)")

    if vix_zscore is not None:
        # zscore > 2 → spike → -20; < -1 → calm → +10
        score += max(-20.0, min(10.0, -10.0 * vix_zscore))
        if vix_zscore >= 2.0:
            drivers.append(f"VIX Z={vix_zscore:+.2f}σ (spike)")

    if usd_krw_pct is not None:
        # 원화 약세 (양수) = risk-off pressure
        if abs(usd_krw_pct) >= 2.0:
            score -= 25
            drivers.append(f"USD/KRW {usd_krw_pct:+.2f}% (FX stress)")
        elif abs(usd_krw_pct) >= 1.0:
            score -= 10
            drivers.append(f"USD/KRW {usd_krw_pct:+.2f}% (moderate FX move)")

    if correlation is not None and correlation < 0.0:
        score -= 10
        drivers.append(f"KOSPI-SPY decoupled (ρ={correlation:.2f})")

    score = max(-100.0, min(100.0, score))

    if score <= -60:
        label = "crisis"
    elif score <= -30:
        label = "risk_off"
    elif score < 30:
        label = "neutral"
    elif score < 60:
        label = "risk_on"
    else:
        label = "euphoria"

    return {
        "label": label,
        "score": round(score, 1),
        "drivers": drivers,
    }


def compute_correlation(
    db_path: Path,
    sym_a: str,
    sym_b: str,
    days: int = 30,
) -> float | None:
    """timeseries.db에서 두 symbol의 일별 가격 Pearson correlation.

    가격이 시간순으로 일치하는 row만 사용. 표준편차 0이면 None.
    """
    if not db_path.exists():
        return None
    sql = (
        "SELECT ts_kst, symbol, price FROM quote_history "
        "WHERE symbol IN (?, ?) AND price IS NOT NULL "
        "ORDER BY ts_kst ASC"
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, (sym_a, sym_b)).fetchall()
    # ts_kst → {sym: price}
    by_ts: dict[str, dict[str, float]] = {}
    for ts, sym, price in rows:
        by_ts.setdefault(ts, {})[sym] = float(price)
    a_vals, b_vals = [], []
    for ts in sorted(by_ts):
        bucket = by_ts[ts]
        if sym_a in bucket and sym_b in bucket:
            a_vals.append(bucket[sym_a])
            b_vals.append(bucket[sym_b])
    if len(a_vals) < 2:
        return None
    try:
        var_a = statistics.pvariance(a_vals)
        var_b = statistics.pvariance(b_vals)
    except statistics.StatisticsError:
        return None
    if var_a == 0 or var_b == 0:
        return None
    mean_a = statistics.fmean(a_vals)
    mean_b = statistics.fmean(b_vals)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a_vals, b_vals)) / len(a_vals)
    denom = math.sqrt(var_a * var_b)
    return round(cov / denom, 4) if denom else None


def _alert_for_transition(prev_label: str, new: dict[str, Any]) -> dict[str, Any] | None:
    """라벨 전환 시 alert dict 생성 (compare.py 포맷 호환)."""
    new_label = new["label"]
    score = new["score"]
    abs_score = abs(score)
    if abs_score >= 70:
        sev = "critical"
    elif abs_score >= 40:
        sev = "high"
    else:
        sev = "medium"
    drivers_str = " · ".join(new["drivers"][:3]) if new["drivers"] else ""
    return {
        "category": "regime",
        "metric": new_label,
        "severity": sev,
        "message": f"Regime 전환 {prev_label} → {new_label} (score {score:+.0f}). {drivers_str}",
        "value": score,
        "threshold": None,
        "delta_from_prev": None,
    }


def detect_regime(
    snapshot: dict[str, Any],
    timeseries_db: Path,
    state_file: Path,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """snapshot + timeseries DB → 라벨 + transition 판정 + alert (전환 시).

    snapshot 구조: pulse.py의 latest.json (indices.vix, fx.usd_krw 등)
    state_file: last_regime.json 경로 — 직전 라벨 비교용.
    """
    vix_q = snapshot.get("indices", {}).get("vix", {})
    fx_q = snapshot.get("fx", {}).get("usd_krw", {})
    vix = vix_q.get("price")
    fx_pct = fx_q.get("pct_change")

    # VIX Z-score (timeseries.db에서 30일)
    vix_z: float | None = None
    if timeseries_db.exists():
        try:
            with sqlite3.connect(timeseries_db) as conn:
                rows = conn.execute(
                    "SELECT price FROM quote_history WHERE symbol='vix' "
                    "AND ts_utc >= datetime('now', ?) "
                    "ORDER BY ts_utc ASC",
                    (f"-{lookback_days} days",),
                ).fetchall()
                vals = [float(r[0]) for r in rows if r[0] is not None]
                if len(vals) >= 3 and vix is not None:
                    mu = statistics.fmean(vals)
                    sd = statistics.stdev(vals)
                    if sd > 0:
                        vix_z = (vix - mu) / sd
        except sqlite3.Error:  # noqa: PERF203
            pass

    correl = compute_correlation(timeseries_db, "kospi", "sp500", days=lookback_days)
    out = label_from_signals(vix=vix, vix_zscore=vix_z, usd_krw_pct=fx_pct, correlation=correl)

    prev_label: str | None = None
    if state_file.exists():
        try:
            prev = json.loads(state_file.read_text())
            prev_label = prev.get("label")
        except (json.JSONDecodeError, OSError):
            prev_label = None

    transition = bool(prev_label and prev_label != out["label"])
    out["transition"] = transition
    out["previous_label"] = prev_label
    out["alert"] = _alert_for_transition(prev_label, out) if transition else None

    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({
        "label": out["label"],
        "score": out["score"],
        "drivers": out["drivers"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2))

    return out
