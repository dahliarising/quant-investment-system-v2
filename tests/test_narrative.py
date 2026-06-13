"""Tests for corvin_jarvis.narrative readonly adapter."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corvin_jarvis import narrative

SCHEMA = """
CREATE TABLE signals (
    date TEXT NOT NULL,
    session TEXT NOT NULL,
    vix REAL NOT NULL,
    foreign_net_buy INTEGER NOT NULL,
    sentiment_tone REAL NOT NULL,
    market TEXT NOT NULL DEFAULT 'KR',
    PRIMARY KEY (date, session, market)
);
"""


@pytest.fixture
def fake_signals_db(tmp_path: Path) -> Path:
    db = tmp_path / "signals.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(SCHEMA)
        rows = [
            ("2026-05-19", "AM", 22.0, 100000, -0.5, "KR"),
            ("2026-05-19", "PM", 23.0, -50000, -0.7, "KR"),
            ("2026-05-20", "AM", 22.5, 80000, -0.3, "KR"),
            ("2026-05-20", "PM", 24.0, -120000, -0.9, "KR"),
            ("2026-05-21", "AM", 25.0, 200000, -1.2, "KR"),
        ]
        conn.executemany(
            "INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?)", rows
        )
    return db


@pytest.mark.unit
def test_latest_signal_returns_most_recent_row(fake_signals_db: Path) -> None:
    row = narrative.latest_signal(fake_signals_db, market="KR")
    assert row is not None
    assert row["date"] == "2026-05-21"
    assert row["session"] == "AM"
    assert row["vix"] == 25.0


@pytest.mark.unit
def test_latest_signal_returns_none_for_missing_market(fake_signals_db: Path) -> None:
    row = narrative.latest_signal(fake_signals_db, market="US")
    assert row is None


@pytest.mark.unit
def test_latest_signal_returns_none_for_missing_db(tmp_path: Path) -> None:
    row = narrative.latest_signal(tmp_path / "nope.db", market="KR")
    assert row is None


@pytest.mark.unit
def test_compute_zscore_for_sentiment(fake_signals_db: Path) -> None:
    """5개 sentiment_tone 값: -0.5, -0.7, -0.3, -0.9, -1.2.
       mean=-0.72, stdev≈0.348. Z of latest -1.2 = (-1.2 - (-0.72))/0.348 ≈ -1.38."""
    result = narrative.compute_zscore(
        fake_signals_db, metric="sentiment_tone", lookback_days=30, market="KR"
    )
    assert result is not None
    assert result["metric"] == "sentiment_tone"
    assert result["n"] == 5
    assert result["latest"] == -1.2
    assert -1.5 < result["zscore"] < -1.3


@pytest.mark.unit
def test_compute_zscore_for_foreign_net_buy(fake_signals_db: Path) -> None:
    result = narrative.compute_zscore(
        fake_signals_db, metric="foreign_net_buy", lookback_days=30, market="KR"
    )
    assert result is not None
    assert result["metric"] == "foreign_net_buy"
    assert result["latest"] == 200000


@pytest.mark.unit
def test_compute_zscore_returns_none_when_insufficient(tmp_path: Path) -> None:
    db = tmp_path / "tiny.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO signals VALUES ('2026-05-21', 'AM', 22.0, 100000, -0.5, 'KR')")
    result = narrative.compute_zscore(db, metric="sentiment_tone", lookback_days=30, market="KR")
    assert result is None


@pytest.mark.unit
def test_compute_zscore_rejects_invalid_metric(fake_signals_db: Path) -> None:
    with pytest.raises(ValueError):
        narrative.compute_zscore(fake_signals_db, metric="injection); DROP TABLE--", lookback_days=30)


@pytest.mark.unit
def test_build_narrative_alerts_no_extreme(fake_signals_db: Path) -> None:
    """fixture에선 |Z| ≈ 1.3-1.4, threshold=2.0 → alert 없음."""
    alerts = narrative.build_narrative_alerts(fake_signals_db, threshold=2.0)
    assert alerts == []


@pytest.mark.unit
def test_build_narrative_alerts_triggers_on_extreme(tmp_path: Path) -> None:
    """sentiment_tone에 큰 spike 추가 → |Z|>2σ alert 발생."""
    db = tmp_path / "spike.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(SCHEMA)
        base = [
            ("2026-05-15", "AM", 22.0, 100000, 0.1, "KR"),
            ("2026-05-15", "PM", 22.5, 110000, 0.05, "KR"),
            ("2026-05-16", "AM", 22.2, 120000, 0.0, "KR"),
            ("2026-05-16", "PM", 22.3, 95000, 0.1, "KR"),
            ("2026-05-17", "AM", 22.1, 105000, 0.05, "KR"),
            # 극단 spike (latest)
            ("2026-05-18", "AM", 35.0, -500000, -3.5, "KR"),
        ]
        conn.executemany("INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?)", base)
    alerts = narrative.build_narrative_alerts(db, threshold=2.0)
    assert len(alerts) >= 1
    sentiments = [a for a in alerts if a["metric"] == "sentiment_tone"]
    assert len(sentiments) == 1
    assert sentiments[0]["category"] == "narrative"
    assert sentiments[0]["severity"] in {"high", "critical"}


@pytest.mark.unit
def test_build_narrative_alerts_handles_missing_db(tmp_path: Path) -> None:
    alerts = narrative.build_narrative_alerts(tmp_path / "nope.db", threshold=2.0)
    assert alerts == []


# ── 진입 게이트: 외국인 순매도·감성 Z (2026-06-11) ──────

def test_entry_caution_from_z_foreign_dump():
    """외국인 강한 순매도(Z 음수) → caution + throttle factor."""
    out = narrative.entry_caution_from_z(fnb_z=-2.0, tone_z=0.1)
    assert out["caution"] is True
    assert out["factor"] == 0.7
    assert "외국인" in out["reason"]


def test_entry_caution_from_z_sentiment_negative():
    out = narrative.entry_caution_from_z(fnb_z=0.0, tone_z=-1.8)
    assert out["caution"] is True
    assert "감성" in out["reason"]


def test_entry_caution_from_z_calm_is_neutral():
    out = narrative.entry_caution_from_z(fnb_z=0.3, tone_z=-0.5)
    assert out["caution"] is False
    assert out["factor"] == 1.0


def test_entry_caution_from_z_none_safe():
    out = narrative.entry_caution_from_z(fnb_z=None, tone_z=None)
    assert out["caution"] is False and out["factor"] == 1.0


def test_entry_caution_db_wrapper(tmp_path):
    """DB 래퍼 — 외국인 순매도 추세를 z로 잡아 caution."""
    db = tmp_path / "signals.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE signals (date TEXT, session TEXT, vix REAL, "
                  "foreign_net_buy REAL, sentiment_tone REAL, market TEXT)")
        # 평소 +50, 최근 급락 -300 (강한 순매도 Z)
        for d in range(2, 15):
            c.execute("INSERT INTO signals VALUES (?,?,?,?,?,?)",
                      (f"2026-06-{d:02d}", "close", 18.0, 50.0, 0.1, "KR"))
        c.execute("INSERT INTO signals VALUES (?,?,?,?,?,?)",
                  ("2026-06-15", "close", 22.0, -300.0, 0.1, "KR"))
    out = narrative.entry_caution(db_path=db, market="KR")
    assert out["caution"] is True
