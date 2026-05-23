"""Tests for corvin_jarvis.earnings module."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from corvin_jarvis import earnings


@pytest.mark.unit
def test_init_earnings_table_creates_table(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='earnings_calendar'"
        )
        assert cur.fetchone() is not None


@pytest.mark.unit
def test_init_earnings_table_creates_index(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    with sqlite3.connect(tmp_db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='earnings_calendar'"
        )
        names = {row[0] for row in cur.fetchall()}
    assert "idx_ec_symbol_date" in names


@pytest.mark.unit
def test_init_earnings_table_idempotent(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    earnings.init_earnings_table(tmp_db_path)
    assert tmp_db_path.exists()


@pytest.mark.unit
def test_upsert_earnings_inserts_new_row(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    n = earnings.upsert_earnings(
        tmp_db_path, "META", [date(2026, 7, 30)],
        eps_avg=7.528, revenue_avg=60209152310,
    )
    assert n == 1
    with sqlite3.connect(tmp_db_path) as conn:
        row = conn.execute(
            "SELECT symbol, earnings_date, eps_avg FROM earnings_calendar"
        ).fetchone()
    assert row == ("META", "2026-07-30", 7.528)


@pytest.mark.unit
def test_upsert_earnings_idempotent(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    earnings.upsert_earnings(tmp_db_path, "META", [date(2026, 7, 30)], eps_avg=7.5, revenue_avg=6e10)
    earnings.upsert_earnings(tmp_db_path, "META", [date(2026, 7, 30)], eps_avg=7.6, revenue_avg=6.1e10)
    with sqlite3.connect(tmp_db_path) as conn:
        rows = conn.execute(
            "SELECT eps_avg FROM earnings_calendar WHERE symbol='META' AND earnings_date='2026-07-30'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 7.6  # 두 번째 호출이 update


@pytest.mark.unit
def test_upsert_earnings_handles_multiple_dates(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    n = earnings.upsert_earnings(
        tmp_db_path, "MSFT",
        [date(2026, 7, 24), date(2026, 10, 23)],
        eps_avg=3.2, revenue_avg=7e10,
    )
    assert n == 2


@pytest.mark.unit
def test_pending_earnings_returns_within_window(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    today = date(2026, 5, 23)
    earnings.upsert_earnings(tmp_db_path, "META", [date(2026, 5, 24)], eps_avg=7.5, revenue_avg=6e10)
    earnings.upsert_earnings(tmp_db_path, "MSFT", [date(2026, 5, 26)], eps_avg=3.2, revenue_avg=7e10)
    earnings.upsert_earnings(tmp_db_path, "NVDA", [date(2026, 8, 28)], eps_avg=1.0, revenue_avg=5e10)

    rows = earnings.pending_earnings(tmp_db_path, today=today, days_ahead=7)
    syms = [r["symbol"] for r in rows]
    assert "META" in syms
    assert "MSFT" in syms
    assert "NVDA" not in syms


@pytest.mark.unit
def test_pending_earnings_excludes_past(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    today = date(2026, 5, 23)
    earnings.upsert_earnings(tmp_db_path, "AAPL", [date(2026, 5, 22)], eps_avg=1.5, revenue_avg=9e10)
    rows = earnings.pending_earnings(tmp_db_path, today=today, days_ahead=7)
    assert rows == []


@pytest.mark.unit
def test_fetch_earnings_date_parses_yfinance_calendar() -> None:
    fake_calendar = {
        "Earnings Date": [date(2026, 7, 30)],
        "Earnings Average": 7.528,
        "Revenue Average": 60209152310,
    }
    with patch("corvin_jarvis.earnings._yf_ticker_calendar", return_value=fake_calendar):
        result = earnings.fetch_earnings_date("META")
    assert result == {
        "symbol": "META",
        "dates": [date(2026, 7, 30)],
        "eps_avg": 7.528,
        "revenue_avg": 60209152310,
        "error": None,
    }


@pytest.mark.unit
def test_fetch_earnings_date_handles_missing_fields() -> None:
    fake_calendar = {
        "Earnings Date": [date(2026, 8, 28)],
    }
    with patch("corvin_jarvis.earnings._yf_ticker_calendar", return_value=fake_calendar):
        result = earnings.fetch_earnings_date("NVDA")
    assert result["dates"] == [date(2026, 8, 28)]
    assert result["eps_avg"] is None
    assert result["revenue_avg"] is None
    assert result["error"] is None


@pytest.mark.unit
def test_fetch_earnings_date_handles_empty_calendar() -> None:
    with patch("corvin_jarvis.earnings._yf_ticker_calendar", return_value={}):
        result = earnings.fetch_earnings_date("XXX")
    assert result["dates"] == []
    assert result["error"] is None


@pytest.mark.unit
def test_fetch_earnings_date_handles_exception() -> None:
    with patch("corvin_jarvis.earnings._yf_ticker_calendar", side_effect=RuntimeError("api down")):
        result = earnings.fetch_earnings_date("ZZZ")
    assert result["dates"] == []
    assert "api down" in result["error"]


@pytest.mark.unit
def test_build_earnings_alerts_d_minus_7(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    today = date(2026, 5, 23)
    earnings.upsert_earnings(tmp_db_path, "META", [date(2026, 5, 30)], eps_avg=7.5, revenue_avg=6e10)
    alerts = earnings.build_earnings_alerts(tmp_db_path, today=today)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["category"] == "earnings"
    assert a["metric"] == "META"
    assert a["severity"] == "medium"  # D-7
    assert "D-7" in a["message"]


@pytest.mark.unit
def test_build_earnings_alerts_d_minus_1_high_severity(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    today = date(2026, 5, 23)
    earnings.upsert_earnings(tmp_db_path, "AAPL", [date(2026, 5, 24)], eps_avg=1.5, revenue_avg=9e10)
    alerts = earnings.build_earnings_alerts(tmp_db_path, today=today)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "high"  # D-1
    assert "D-1" in alerts[0]["message"]


@pytest.mark.unit
def test_build_earnings_alerts_skips_non_threshold_days(tmp_db_path: Path) -> None:
    """D-5 같은 비 alert day는 skip."""
    earnings.init_earnings_table(tmp_db_path)
    today = date(2026, 5, 23)
    earnings.upsert_earnings(tmp_db_path, "NVDA", [date(2026, 5, 28)], eps_avg=1.0, revenue_avg=5e10)
    alerts = earnings.build_earnings_alerts(tmp_db_path, today=today)
    assert alerts == []


@pytest.mark.unit
def test_build_earnings_alerts_multiple_symbols(tmp_db_path: Path) -> None:
    earnings.init_earnings_table(tmp_db_path)
    today = date(2026, 5, 23)
    earnings.upsert_earnings(tmp_db_path, "META", [date(2026, 5, 30)], eps_avg=7.5, revenue_avg=6e10)
    earnings.upsert_earnings(tmp_db_path, "AAPL", [date(2026, 5, 26)], eps_avg=1.5, revenue_avg=9e10)
    earnings.upsert_earnings(tmp_db_path, "MSFT", [date(2026, 5, 24)], eps_avg=3.2, revenue_avg=7e10)
    alerts = earnings.build_earnings_alerts(tmp_db_path, today=today)
    metrics = {a["metric"]: a["severity"] for a in alerts}
    assert metrics == {"META": "medium", "AAPL": "high", "MSFT": "high"}
