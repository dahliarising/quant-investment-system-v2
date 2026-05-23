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


@pytest.mark.unit
def test_relative_strength_outperform(tmp_db_path: Path) -> None:
    """META +5%, SP500 +2% → RS = +3pp"""
    _seed_history(tmp_db_path, "META", [600.0, 610.0, 615.0, 620.0, 630.0])
    _seed_history(tmp_db_path, "sp500", [5000.0, 5050.0, 5070.0, 5080.0, 5100.0], category="index")
    rs = qa.relative_strength(tmp_db_path, symbol="META", benchmark="sp500", days=10)
    assert rs["symbol_pct"] == pytest.approx(5.0, abs=0.01)
    assert rs["benchmark_pct"] == pytest.approx(2.0, abs=0.01)
    assert rs["relative_pp"] == pytest.approx(3.0, abs=0.01)
    assert rs["verdict"] == "outperform"


@pytest.mark.unit
def test_relative_strength_underperform(tmp_db_path: Path) -> None:
    """META -2%, SP500 +1% → RS = -3pp"""
    _seed_history(tmp_db_path, "META", [600.0, 595.0, 590.0, 595.0, 588.0])
    _seed_history(tmp_db_path, "sp500", [5000.0, 5025.0, 5040.0, 5050.0, 5050.0], category="index")
    rs = qa.relative_strength(tmp_db_path, symbol="META", benchmark="sp500", days=10)
    assert rs["verdict"] == "underperform"
    assert rs["relative_pp"] < 0


@pytest.mark.unit
def test_relative_strength_handles_missing_data(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    rs = qa.relative_strength(tmp_db_path, symbol="X", benchmark="Y", days=10)
    assert rs["verdict"] == "unknown"
    assert rs["symbol_pct"] is None
