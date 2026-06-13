# tests/test_prediction_backfill.py
import sqlite3
from corvin_jarvis.prediction import backfill


def test_init_and_upsert_and_read(tmp_path):
    db = tmp_path / "daily.db"
    backfill.init_db(db)
    rows = [
        {"symbol": "kospi", "date": "2026-06-10", "open": 1, "high": 2, "low": 0.5,
         "close": 1.5, "volume": 100, "source": "fdr"},
        {"symbol": "kospi", "date": "2026-06-11", "open": 1.5, "high": 2.5, "low": 1,
         "close": 2.0, "volume": 120, "source": "fdr"},
    ]
    backfill.upsert_rows(db, rows)
    # 중복 date upsert → 갱신, 행 증가 없음
    backfill.upsert_rows(db, [dict(rows[1], close=9.9)])
    got = backfill.read_daily(db, "kospi", lookback=10)
    assert [r["date"] for r in got] == ["2026-06-10", "2026-06-11"]  # 시간순
    assert got[-1]["close"] == 9.9  # upsert로 갱신됨


def test_last_date_returns_none_when_empty(tmp_path):
    db = tmp_path / "daily.db"
    backfill.init_db(db)
    assert backfill.last_date(db, "kospi") is None
