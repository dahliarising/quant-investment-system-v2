"""Corvin Jarvis — Narrative DB Adapter (Tier 1.4)

narrative-shift-detector의 signals.db readonly 어댑터.

스키마 (signals 테이블):
    date, session, vix, foreign_net_buy, sentiment_tone, market

KR 시장 레벨만 제공 (종목별 narrative는 미지원). |Z| > 2σ alert 생성.

⚠️ readonly only — 절대 write 금지.
"""
from __future__ import annotations

import logging
import sqlite3
import statistics
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.narrative")

DEFAULT_SIGNALS_DB = Path("/Users/thethethe/Claude/narrative-shift-detector/data/signals.db")

ALLOWED_METRICS = {"vix", "foreign_net_buy", "sentiment_tone"}


def _readonly_connect(db_path: Path) -> sqlite3.Connection | None:
    """signals.db readonly 모드로 connect. URI scheme 사용."""
    if not db_path.exists():
        return None
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def compute_zscore(
    db_path: Path,
    metric: str,
    lookback_days: int = 30,
    market: str = "KR",
) -> dict[str, Any] | None:
    """최근 lookback_days 행의 metric Z-score (most recent vs lookback mean/std).

    반환: {"metric", "latest", "mean", "stdev", "zscore", "n"} 또는 None (n<2).
    """
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"invalid metric: {metric!r}. allowed={sorted(ALLOWED_METRICS)}")
    conn = _readonly_connect(db_path)
    if conn is None:
        return None
    try:
        sql = (
            f"SELECT date, session, {metric} AS value FROM signals "
            "WHERE market = ? AND date >= date('now', ? || ' days') "
            "ORDER BY date DESC, session DESC"
        )
        rows = conn.execute(sql, (market, f"-{lookback_days}")).fetchall()
        if len(rows) < 2:
            return None
        values = [float(r[2]) for r in rows]
        latest = values[0]
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
        z = (latest - mean) / stdev if stdev > 0 else 0.0
        return {
            "metric": metric,
            "latest": latest,
            "mean": mean,
            "stdev": stdev,
            "zscore": z,
            "n": len(values),
        }
    finally:
        conn.close()


def _severity_from_z(z: float) -> str:
    az = abs(z)
    if az >= 3.0:
        return "critical"
    if az >= 2.0:
        return "high"
    if az >= 1.5:
        return "medium"
    return "low"


def build_narrative_alerts(
    db_path: Path,
    threshold: float = 2.0,
    lookback_days: int = 30,
    market: str = "KR",
) -> list[dict[str, Any]]:
    """sentiment_tone과 foreign_net_buy의 |Z| > threshold 이면 alert."""
    alerts: list[dict[str, Any]] = []
    for metric in ("sentiment_tone", "foreign_net_buy"):
        z = compute_zscore(db_path, metric=metric, lookback_days=lookback_days, market=market)
        if z is None:
            continue
        if abs(z["zscore"]) < threshold:
            continue
        direction = "📈 spike" if z["zscore"] > 0 else "📉 plunge"
        alerts.append({
            "category": "narrative",
            "metric": metric,
            "severity": _severity_from_z(z["zscore"]),
            "message": (
                f"{market} {metric} {direction} "
                f"Z={z['zscore']:+.2f}σ (latest={z['latest']}, "
                f"μ={z['mean']:.3f}, σ={z['stdev']:.3f}, n={z['n']})"
            ),
            "value": round(z["zscore"], 3),
            "threshold": threshold,
            "delta_from_prev": None,
        })
    return alerts


def latest_signal(db_path: Path, market: str = "KR") -> dict[str, Any] | None:
    """가장 최근 (date DESC, session DESC) 행 반환. 없으면 None."""
    conn = _readonly_connect(db_path)
    if conn is None:
        return None
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT date, session, vix, foreign_net_buy, sentiment_tone, market "
            "FROM signals WHERE market = ? "
            "ORDER BY date DESC, session DESC LIMIT 1",
            (market,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── 진입 게이트: 외국인 순매도·감성 악화 → 신규 진입 throttle (KR) ──

_NARR_Z_THRESHOLD = 1.5     # |Z| 이 임계 넘으면 caution
_NARR_THROTTLE_FACTOR = 0.7  # caution 시 DCA 점수 배율


def entry_caution_from_z(fnb_z: float | None, tone_z: float | None,
                         threshold: float = _NARR_Z_THRESHOLD) -> dict[str, Any]:
    """순수 판정: 외국인 순매도 Z·감성 Z → 진입 주의 (KR narrative).

    fnb_z ≤ -threshold (외국인 평소보다 강한 순매도) 또는
    tone_z ≤ -threshold (감성 악화) → caution(factor<1).
    """
    reasons: list[str] = []
    if fnb_z is not None and fnb_z <= -threshold:
        reasons.append(f"외국인 순매도 Z{fnb_z:+.1f}")
    if tone_z is not None and tone_z <= -threshold:
        reasons.append(f"감성 악화 Z{tone_z:+.1f}")
    return {"caution": bool(reasons),
            "factor": _NARR_THROTTLE_FACTOR if reasons else 1.0,
            "reason": " · ".join(reasons)}


def entry_caution(db_path: Path | None = None, market: str = "KR",
                  lookback_days: int = 30) -> dict[str, Any]:
    """DB 래퍼 — 최근 narrative z-score로 진입 주의 판정. DB 없으면 factor 1.0."""
    p = db_path or DEFAULT_SIGNALS_DB
    try:
        fnb = compute_zscore(p, "foreign_net_buy", lookback_days, market)
        tone = compute_zscore(p, "sentiment_tone", lookback_days, market)
    except (ValueError, sqlite3.Error):
        return {"caution": False, "factor": 1.0, "reason": ""}
    return entry_caution_from_z(fnb["zscore"] if fnb else None,
                                tone["zscore"] if tone else None)
