"""signals/calibration.py — 적중률 집계 + 베이지안 수축 보정."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from corvin_jarvis.signals import calibration, ledger

KST = ZoneInfo("Asia/Seoul")
NOW = datetime(2026, 6, 10, 16, 30, tzinfo=KST)


def _seed(db, n_hit, n_miss, n_late=0, engine="predictive", kind="VELOCITY"):
    """채점 완료된 신호 n건 시드."""
    for i in range(n_hit + n_miss + n_late):
        ledger.record_batch(engine, [{
            "symbol": f"SYM{i}", "kind": kind, "confidence": 65.0,
            "horizon_days": 5, "stop": 100.0,
        }], db_path=db, now=NOW)
    import sqlite3
    with sqlite3.connect(db) as conn:
        rows = [r[0] for r in conn.execute("SELECT id FROM signal_ledger ORDER BY id").fetchall()]
    statuses = ["hit"] * n_hit + ["late_hit"] * n_late + ["miss"] * n_miss
    for rid, st in zip(rows, statuses):
        ledger.mark_scored(rid, st, {}, db_path=db, now=NOW)


def test_compute_hit_rate_with_late_half_weight(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=6, n_miss=2, n_late=2)  # (6 + 0.5*2)/10 = 0.70
    stats = calibration.compute(db_path=db)
    entry = stats["predictive"]["VELOCITY"]
    assert entry["n"] == 10
    assert abs(entry["hit_rate"] - 0.70) < 1e-9


def test_small_sample_below_10_not_calibrated(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=3, n_miss=2)  # n=5 < 10
    stats = calibration.compute(db_path=db)
    entry = stats["predictive"]["VELOCITY"]
    assert entry["calibrated_confidence"] is None  # 보정 보류


def test_calibrated_confidence_shrinks_toward_hit_rate(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=4, n_miss=6)  # hit_rate 0.40, n=10 → shrink 0.7
    stats = calibration.compute(db_path=db)
    entry = stats["predictive"]["VELOCITY"]
    # 65*0.7 + 40*0.3 = 57.5
    assert abs(entry["calibrated_confidence"] - 57.5) < 0.1


def test_unscorable_excluded_from_denominator(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=10, n_miss=0)
    ledger.record_batch("predictive", [{
        "symbol": "ZZZ", "kind": "VELOCITY", "confidence": 65.0,
        "horizon_days": 5,
    }], db_path=db, now=NOW)
    import sqlite3
    with sqlite3.connect(db) as conn:
        rid = conn.execute("SELECT MAX(id) FROM signal_ledger").fetchone()[0]
    ledger.mark_scored(rid, "unscorable", {}, db_path=db, now=NOW)
    stats = calibration.compute(db_path=db)
    assert stats["predictive"]["VELOCITY"]["n"] == 10  # unscorable 제외


def test_write_state_creates_json(tmp_path):
    db = tmp_path / "ledger.db"
    out = tmp_path / "calibration.json"
    _seed(db, n_hit=8, n_miss=2)
    calibration.write_state(db_path=db, out_path=out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "predictive" in data
    assert data["predictive"]["VELOCITY"]["n"] == 10


def test_scoreboard_rows_for_dashboard(tmp_path):
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=7, n_miss=3)
    rows = calibration.scoreboard(db_path=db)
    assert len(rows) == 1
    r = rows[0]
    assert r["engine"] == "predictive"
    assert r["kind"] == "VELOCITY"
    assert r["n"] == 10
    assert r["hit_rate"] == 0.70
    assert r["open"] == 0


def test_scoreboard_shows_open_only_pairs(tmp_path):
    """채점 이력 없는 open-only 페어 — 부트스트랩 기간에도 행 표시."""
    db = tmp_path / "ledger.db"
    ledger.record_batch("signal_engine", [{
        "symbol": "NVDA", "kind": "WATCH", "confidence": None, "horizon_days": 5,
    }], db_path=db, now=NOW)
    rows = calibration.scoreboard(db_path=db)
    assert len(rows) == 1
    assert rows[0]["engine"] == "signal_engine"
    assert rows[0]["kind"] == "WATCH"
    assert rows[0]["n"] == 0
    assert rows[0]["hit_rate"] is None
    assert rows[0]["open"] == 1


def test_no_calibration_when_all_confidence_null(tmp_path):
    """confidence 미기록 엔진 — n 충분해도 보정 보류 (None)."""
    db = tmp_path / "ledger.db"
    _seed(db, n_hit=8, n_miss=2)
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE signal_ledger SET confidence=NULL")
    stats = calibration.compute(db_path=db)
    assert stats["predictive"]["VELOCITY"]["calibrated_confidence"] is None
    assert stats["predictive"]["VELOCITY"]["n"] == 10
