"""Tests for corvin_jarvis.qa Q&A retrieval toolkit."""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis import qa, timeseries


def _seed_history(db_path: Path, symbol: str, prices: list[float], category: str = "portfolio") -> None:
    """헬퍼: timeseries.db에 N일치 가짜 가격 row 삽입."""
    timeseries.init_db(db_path)
    for i, p in enumerate(prices):
        ts = f"2026-05-{15+i:02d}T00:00:00+00:00"
        kst = f"2026-05-{15+i:02d}T09:00:00+09:00"
        snap = {
            "timestamp_utc": ts,
            "timestamp_kst": kst,
            "indices": {},
            "commodities": {},
            "fx": {},
            "portfolio": [],
            "watchlist": [],
        }
        if category == "portfolio":
            snap["portfolio"] = [{
                "symbol": symbol, "shares": 1, "avg_price": p, "currency": "USD",
                "current_price": p, "pnl_pct": 0.0, "market_value": p, "error": None,
            }]
        elif category == "index":
            snap["indices"] = {symbol: {"price": p, "pct_change": 0.0, "source": "test", "error": None}}
        timeseries.write_snapshot(db_path, snap)


@pytest.mark.unit
def test_recent_history_summary_basic(tmp_db_path: Path) -> None:
    _seed_history(tmp_db_path, "META", [600.0, 605.0, 595.0, 610.0, 605.0])
    summary = qa.recent_history_summary(tmp_db_path, symbol="META", days=10)
    assert summary["symbol"] == "META"
    assert summary["n"] == 5
    assert summary["start_price"] == 600.0
    assert summary["end_price"] == 605.0
    assert summary["min_price"] == 595.0
    assert summary["max_price"] == 610.0
    assert summary["pct_change"] == pytest.approx((605 / 600 - 1) * 100, abs=0.01)


@pytest.mark.unit
def test_recent_history_summary_missing_symbol(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    summary = qa.recent_history_summary(tmp_db_path, symbol="UNKNOWN", days=10)
    assert summary["n"] == 0
    assert summary["start_price"] is None
