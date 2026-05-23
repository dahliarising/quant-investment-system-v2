"""Tests for corvin_jarvis.timeseries module."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corvin_jarvis import timeseries


@pytest.mark.unit
def test_init_db_creates_quote_history_table(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='quote_history'"
        )
        assert cur.fetchone() is not None


@pytest.mark.unit
def test_init_db_creates_indexes(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='quote_history'"
        )
        names = {row[0] for row in cur.fetchall()}
    assert "idx_qh_symbol_ts" in names
    assert "idx_qh_category_ts" in names


@pytest.mark.unit
def test_init_db_is_idempotent(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    timeseries.init_db(tmp_db_path)
    assert tmp_db_path.exists()


@pytest.mark.unit
def test_write_snapshot_inserts_index_rows(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    snapshot = {
        "timestamp_utc": "2026-05-21T00:00:00+00:00",
        "timestamp_kst": "2026-05-21T09:00:00+09:00",
        "indices": {
            "kospi": {"price": 7208.95, "pct_change": -0.86, "source": "FDR", "error": None},
            "sp500": {"price": 7432.97, "pct_change": 1.08, "source": "yfinance", "error": None},
        },
        "commodities": {},
        "fx": {},
        "portfolio": [],
        "watchlist": [],
    }
    timeseries.write_snapshot(tmp_db_path, snapshot)
    with sqlite3.connect(tmp_db_path) as conn:
        rows = conn.execute(
            "SELECT category, symbol, price, pct_change FROM quote_history ORDER BY symbol"
        ).fetchall()
    assert rows == [
        ("index", "kospi", 7208.95, -0.86),
        ("index", "sp500", 7432.97, 1.08),
    ]


@pytest.mark.unit
def test_write_snapshot_inserts_portfolio_with_pnl(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    snapshot = {
        "timestamp_utc": "2026-05-21T00:00:00+00:00",
        "timestamp_kst": "2026-05-21T09:00:00+09:00",
        "indices": {},
        "commodities": {},
        "fx": {},
        "portfolio": [
            {
                "symbol": "META", "shares": 7, "avg_price": 597.61, "currency": "USD",
                "current_price": 605.06, "pnl_pct": 1.25, "market_value": 4235.42,
                "error": None,
            }
        ],
        "watchlist": [],
    }
    timeseries.write_snapshot(tmp_db_path, snapshot)
    with sqlite3.connect(tmp_db_path) as conn:
        row = conn.execute(
            "SELECT category, symbol, price, pnl_pct, market_value, shares FROM quote_history"
        ).fetchone()
    assert row == ("portfolio", "META", 605.06, 1.25, 4235.42, 7.0)


@pytest.mark.unit
def test_write_snapshot_handles_watchlist(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    snapshot = {
        "timestamp_utc": "2026-05-21T00:00:00+00:00",
        "timestamp_kst": "2026-05-21T09:00:00+09:00",
        "indices": {}, "commodities": {}, "fx": {}, "portfolio": [],
        "watchlist": [
            {"symbol": "TSLA", "price": 350.0, "pct_change": 2.5, "source": "yfinance", "error": None}
        ],
    }
    timeseries.write_snapshot(tmp_db_path, snapshot)
    with sqlite3.connect(tmp_db_path) as conn:
        row = conn.execute(
            "SELECT category, symbol, price, pct_change FROM quote_history"
        ).fetchone()
    assert row == ("watchlist", "TSLA", 350.0, 2.5)


@pytest.mark.unit
def test_write_snapshot_skips_null_price(tmp_db_path: Path) -> None:
    """price가 None이면 row 자체를 skip (error 케이스)."""
    timeseries.init_db(tmp_db_path)
    snapshot = {
        "timestamp_utc": "2026-05-21T00:00:00+00:00",
        "timestamp_kst": "2026-05-21T09:00:00+09:00",
        "indices": {"vix": {"price": None, "pct_change": None, "source": "yfinance", "error": "rate limit"}},
        "commodities": {}, "fx": {}, "portfolio": [], "watchlist": [],
    }
    timeseries.write_snapshot(tmp_db_path, snapshot)
    with sqlite3.connect(tmp_db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM quote_history").fetchone()[0]
    assert count == 0


@pytest.mark.unit
def test_read_history_returns_rows_for_symbol(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    for hour in range(3):
        snapshot = {
            "timestamp_utc": f"2026-05-21T0{hour}:00:00+00:00",
            "timestamp_kst": f"2026-05-21T0{hour+9}:00:00+09:00",
            "indices": {"kospi": {"price": 7200.0 + hour, "pct_change": -0.5, "source": "FDR", "error": None}},
            "commodities": {}, "fx": {}, "portfolio": [], "watchlist": [],
        }
        timeseries.write_snapshot(tmp_db_path, snapshot)
    rows = timeseries.read_history(tmp_db_path, symbol="kospi", limit=10)
    assert len(rows) == 3
    assert [r["price"] for r in rows] == [7200.0, 7201.0, 7202.0]


@pytest.mark.unit
def test_read_history_filter_by_category(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    snapshot = {
        "timestamp_utc": "2026-05-21T00:00:00+00:00",
        "timestamp_kst": "2026-05-21T09:00:00+09:00",
        "indices": {"kospi": {"price": 7200.0, "pct_change": -0.5, "source": "FDR", "error": None}},
        "commodities": {"gold": {"price": 3200.0, "pct_change": 0.5, "source": "yfinance", "error": None}},
        "fx": {}, "portfolio": [], "watchlist": [],
    }
    timeseries.write_snapshot(tmp_db_path, snapshot)
    rows = timeseries.read_history(tmp_db_path, category="commodity", limit=10)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "gold"
