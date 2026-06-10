"""적중률 집계 → confidence 보정 (스펙 §4.3).

hit_rate = (hit + 0.5*late_hit) / (hit+late_hit+miss). unscorable 분모 제외.
표본 n<10 → 보정 보류 (calibrated_confidence=None, 원본 유지).
베이지안 수축: calibrated = avg_conf*shrink + hit_rate*100*(1-shrink).
shrink = n=10→0.7에서 n≥50→0.2로 선형 감소 — 소표본 과보정 방지.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from corvin_jarvis.signals import ledger

STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "calibration.json"
_MIN_SAMPLES = 10


def _shrink(n: int) -> float:
    if n >= 50:
        return 0.2
    return 0.7 - 0.5 * (n - _MIN_SAMPLES) / 40  # n=10→0.7, n=50→0.2 선형


def compute(db_path: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """엔진×kind별 {n, hit_rate, avg_confidence, calibrated_confidence}."""
    p = ledger.init_db(db_path)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    with sqlite3.connect(p) as conn:
        rows = conn.execute(
            """SELECT engine, kind,
                      SUM(status='hit'), SUM(status='late_hit'), SUM(status='miss'),
                      AVG(confidence)
               FROM signal_ledger
               WHERE status IN ('hit','late_hit','miss')
               GROUP BY engine, kind""").fetchall()
    for engine, kind, hits, lates, misses, avg_conf in rows:
        n = (hits or 0) + (lates or 0) + (misses or 0)
        if n == 0:
            continue
        hit_rate = ((hits or 0) + 0.5 * (lates or 0)) / n
        calibrated = None
        if n >= _MIN_SAMPLES and avg_conf is not None:
            s = _shrink(n)
            calibrated = round(avg_conf * s + hit_rate * 100 * (1 - s), 1)
        out.setdefault(engine, {})[kind] = {
            "n": n, "hit_rate": round(hit_rate, 4),
            "avg_confidence": round(avg_conf, 1) if avg_conf is not None else None,
            "calibrated_confidence": calibrated,
        }
    return out


def write_state(db_path: Path | None = None, out_path: Path | None = None) -> Path:
    """compute 결과를 state/calibration.json에 저장 (Phase 3 주입용)."""
    target = out_path or STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(compute(db_path), ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def scoreboard(db_path: Path | None = None) -> list[dict[str, Any]]:
    """대시보드용 행 — open 신호 수 포함, n 내림차순."""
    p = ledger.init_db(db_path)
    stats = compute(db_path)
    with sqlite3.connect(p) as conn:
        open_counts = dict(conn.execute(
            "SELECT engine || '|' || kind, COUNT(*) FROM signal_ledger"
            " WHERE status='open' GROUP BY engine, kind").fetchall())
    rows = []
    for engine, kinds in stats.items():
        for kind, e in kinds.items():
            rows.append({"engine": engine, "kind": kind, "n": e["n"],
                         "hit_rate": round(e["hit_rate"], 2),
                         "calibrated_confidence": e["calibrated_confidence"],
                         "open": open_counts.get(f"{engine}|{kind}", 0)})
    return sorted(rows, key=lambda r: -r["n"])
