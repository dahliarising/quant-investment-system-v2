# tests/test_prediction_backfill_fetch.py
from corvin_jarvis.prediction import backfill


def test_incremental_update_only_fetches_after_last_date(tmp_path):
    db = tmp_path / "daily.db"
    backfill.init_db(db)
    backfill.upsert_rows(db, [{"symbol": "kospi", "date": "2026-06-10", "open": 1,
        "high": 1, "low": 1, "close": 1, "volume": 0, "source": "seed"}])

    calls = {}
    def fake_fetch(symbol, market, start):
        calls[symbol] = start
        return [{"symbol": symbol, "date": "2026-06-11", "open": 2, "high": 2,
                 "low": 2, "close": 2, "volume": 0, "source": "fake"}]

    n = backfill.incremental_update(db, [("kospi", "KR")], fetcher=fake_fetch)
    assert calls["kospi"] == "2026-06-11"      # last_date + 1일부터
    assert n >= 1
    got = backfill.read_daily(db, "kospi", lookback=10)
    assert [r["date"] for r in got] == ["2026-06-10", "2026-06-11"]
