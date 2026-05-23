"""Tests for corvin_jarvis.debate — Tier 4.1 multi-agent context layer."""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis import debate, timeseries


def _seed(db_path: Path, symbol: str, prices: list[float], category: str = "portfolio") -> None:
    timeseries.init_db(db_path)
    for i, p in enumerate(prices):
        snap = {
            "timestamp_utc": f"2026-05-{10+i:02d}T00:00:00+00:00",
            "timestamp_kst": f"2026-05-{10+i:02d}T09:00:00+09:00",
            "indices": {}, "commodities": {}, "fx": {}, "watchlist": [],
            "portfolio": [],
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
def test_bull_context_outperform(tmp_db_path: Path) -> None:
    """META +5%, SP500 +2% → bull signals 포함."""
    _seed(tmp_db_path, "META", [600.0, 610.0, 615.0, 620.0, 630.0])
    _seed(tmp_db_path, "sp500", [5000.0, 5050.0, 5070.0, 5080.0, 5100.0], category="index")
    ctx = debate.gather_bull_context(tmp_db_path, symbol="META", benchmark="sp500")
    assert ctx["symbol"] == "META"
    signals = ctx["signals"]
    assert any("outperform" in s.lower() or "벤치마크" in s for s in signals)


@pytest.mark.unit
def test_bear_context_underperform(tmp_db_path: Path) -> None:
    _seed(tmp_db_path, "META", [600.0, 595.0, 590.0, 595.0, 588.0])
    _seed(tmp_db_path, "sp500", [5000.0, 5025.0, 5040.0, 5050.0, 5050.0], category="index")
    ctx = debate.gather_bear_context(tmp_db_path, symbol="META", benchmark="sp500")
    signals = ctx["signals"]
    assert any("underperform" in s.lower() or "약세" in s or "decline" in s.lower() for s in signals)


@pytest.mark.unit
def test_bull_context_includes_momentum_when_uptrend(tmp_db_path: Path) -> None:
    _seed(tmp_db_path, "META", [600.0, 605.0, 610.0, 615.0, 620.0])
    ctx = debate.gather_bull_context(tmp_db_path, symbol="META", benchmark="sp500")
    assert any("upward" in s.lower() or "상승" in s or "momentum" in s.lower() for s in ctx["signals"])


@pytest.mark.unit
def test_bear_context_includes_downtrend(tmp_db_path: Path) -> None:
    _seed(tmp_db_path, "META", [620.0, 615.0, 610.0, 605.0, 600.0])
    ctx = debate.gather_bear_context(tmp_db_path, symbol="META", benchmark="sp500")
    assert any("downward" in s.lower() or "하락" in s or "downtrend" in s.lower() for s in ctx["signals"])


@pytest.mark.unit
def test_debate_brief_contains_both_sides(tmp_db_path: Path) -> None:
    _seed(tmp_db_path, "META", [600.0, 605.0, 615.0])
    _seed(tmp_db_path, "sp500", [5000.0, 5050.0, 5100.0], category="index")
    brief = debate.debate_brief(tmp_db_path, symbol="META", benchmark="sp500")
    assert "bull" in brief
    assert "bear" in brief
    assert brief["symbol"] == "META"
    assert isinstance(brief["bull"]["signals"], list)
    assert isinstance(brief["bear"]["signals"], list)


@pytest.mark.unit
def test_bull_bear_handle_no_data(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    bull = debate.gather_bull_context(tmp_db_path, "ZZZ", benchmark="sp500")
    bear = debate.gather_bear_context(tmp_db_path, "ZZZ", benchmark="sp500")
    assert bull["signals"] == [] or all("데이터" in s or "data" in s.lower() for s in bull["signals"])
    assert bear["signals"] == [] or all("데이터" in s or "data" in s.lower() for s in bear["signals"])
