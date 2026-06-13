# tests/test_prediction_feeds.py
"""narrative-shift-detector signals.db → 예측 모듈 payload 배선."""
import sqlite3

from corvin_jarvis.prediction import feeds

_SCHEMA = """CREATE TABLE signals (
    date TEXT NOT NULL, session TEXT NOT NULL, vix REAL NOT NULL,
    foreign_net_buy INTEGER NOT NULL, sentiment_tone REAL NOT NULL,
    market TEXT NOT NULL DEFAULT 'KR', PRIMARY KEY (date, session, market))"""


def _make_db(tmp_path, rows):
    db = tmp_path / "signals.db"
    with sqlite3.connect(db) as c:
        c.execute(_SCHEMA)
        c.executemany("INSERT INTO signals VALUES (?,?,?,?,?,?)", rows)
    return db


def test_missing_db_returns_none(tmp_path):
    assert feeds.fetch_sentiment(tmp_path / "nope.db") is None


def test_empty_db_returns_none(tmp_path):
    db = _make_db(tmp_path, [])
    assert feeds.fetch_sentiment(db) is None


def test_latest_tone_and_rising_trend(tmp_path):
    rows = [
        ("2026-06-10", "close", 18.0, 1000, -0.5, "KR"),
        ("2026-06-11", "close", 18.0, 1000, -0.2, "KR"),
        ("2026-06-13", "premarket", 17.0, 1000, 0.4, "KR"),  # 최신, 평균보다 높음
    ]
    db = _make_db(tmp_path, rows)
    p = feeds.fetch_sentiment(db)
    assert p is not None
    assert abs(p["tone"] - 0.4) < 1e-9       # 최신 날짜 tone
    assert p["trend"] == "개선"               # 최신 > 최근 평균


def test_payload_consumable_by_m_sentiment(tmp_path):
    from corvin_jarvis.prediction import m_sentiment
    db = _make_db(tmp_path, [("2026-06-13", "close", 17.0, 1000, 0.9, "KR")])
    p = feeds.fetch_sentiment(db)
    r = m_sentiment.run(p)
    assert r.data_ok is True
    assert r.evidence["tone"] == 0.9
