"""Tests for corvin_jarvis.predict — predictive alerts."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from corvin_jarvis import predict, timeseries


def _seed_prices(db_path: Path, symbol: str, prices: list[float], category: str = "portfolio") -> None:
    timeseries.init_db(db_path)
    for i, p in enumerate(prices):
        ts = f"2026-05-{10+i:02d}T00:00:00+00:00"
        kst = f"2026-05-{10+i:02d}T09:00:00+09:00"
        snap = {
            "timestamp_utc": ts, "timestamp_kst": kst,
            "indices": {}, "commodities": {}, "fx": {},
            "portfolio": [], "watchlist": [],
        }
        if category == "portfolio":
            snap["portfolio"] = [{
                "symbol": symbol, "shares": 1, "avg_price": p, "currency": "USD",
                "current_price": p, "pnl_pct": 0.0, "market_value": p, "error": None,
            }]
        else:
            snap["indices"] = {symbol: {"price": p, "pct_change": 0.0, "source": "test", "error": None}}
        timeseries.write_snapshot(db_path, snap)


@pytest.mark.unit
def test_log_returns_basic(tmp_db_path: Path) -> None:
    _seed_prices(tmp_db_path, "META", [600.0, 606.0, 612.0])
    rs = predict.log_returns(tmp_db_path, "META", days=10)
    assert len(rs) == 2
    assert rs[0] == pytest.approx(math.log(606 / 600), abs=1e-6)
    assert rs[1] == pytest.approx(math.log(612 / 606), abs=1e-6)


@pytest.mark.unit
def test_log_returns_empty_when_no_data(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    assert predict.log_returns(tmp_db_path, "META", days=10) == []


@pytest.mark.unit
def test_log_returns_skip_zero_or_negative(tmp_db_path: Path) -> None:
    """가격에 None이나 0이 섞이면 skip."""
    timeseries.init_db(tmp_db_path)
    for i, p in enumerate([600.0, 0.0, 612.0]):
        snap = {
            "timestamp_utc": f"2026-05-{10+i:02d}T00:00:00+00:00",
            "timestamp_kst": f"2026-05-{10+i:02d}T09:00:00+09:00",
            "indices": {},
            "commodities": {},
            "fx": {},
            "portfolio": [{
                "symbol": "X", "shares": 1, "avg_price": 1, "currency": "USD",
                "current_price": p, "pnl_pct": 0, "market_value": p, "error": None,
            }] if p > 0 else [],
            "watchlist": [],
        }
        timeseries.write_snapshot(tmp_db_path, snap)
    rs = predict.log_returns(tmp_db_path, "X", days=10)
    assert all(math.isfinite(r) for r in rs)
