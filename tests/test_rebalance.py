"""Tests for corvin_jarvis.rebalance."""
from __future__ import annotations

import pytest

from corvin_jarvis import rebalance


def _h(symbol: str, shares: float, currency: str = "USD") -> dict:
    return {"symbol": symbol, "shares": shares,
            "avgPriceUSD": 100.0, "avgPriceKRW": 150000, "currency": currency}


@pytest.mark.unit
def test_detect_drift_within_tolerance_returns_empty() -> None:
    targets = {"META": 0.5, "MSFT": 0.5}
    holdings = [_h("META", 50), _h("MSFT", 50)]
    prices = {"META": 100.0, "MSFT": 100.0}
    drifts = rebalance.detect_drift(holdings, targets, prices, tolerance=0.05)
    assert drifts == []


@pytest.mark.unit
def test_detect_drift_overweight_triggers() -> None:
    """META 60%, MSFT 40% → target 50/50 → META +10pp, MSFT -10pp drift."""
    targets = {"META": 0.5, "MSFT": 0.5}
    holdings = [_h("META", 60), _h("MSFT", 40)]
    prices = {"META": 100.0, "MSFT": 100.0}
    drifts = rebalance.detect_drift(holdings, targets, prices, tolerance=0.05)
    assert len(drifts) == 2
    by_sym = {d["symbol"]: d for d in drifts}
    assert by_sym["META"]["drift_pp"] == pytest.approx(10.0, abs=0.1)
    assert by_sym["META"]["direction"] == "overweight"
    assert by_sym["MSFT"]["direction"] == "underweight"


@pytest.mark.unit
def test_detect_drift_missing_target_position_is_underweight() -> None:
    """target에 있는데 holdings에 없으면 underweight = target%."""
    targets = {"META": 0.5, "NEW": 0.5}
    holdings = [_h("META", 100)]
    prices = {"META": 100.0, "NEW": 50.0}
    drifts = rebalance.detect_drift(holdings, targets, prices, tolerance=0.05)
    by_sym = {d["symbol"]: d for d in drifts}
    assert by_sym["NEW"]["direction"] == "underweight"
    assert by_sym["NEW"]["drift_pp"] == pytest.approx(-50.0, abs=0.1)


@pytest.mark.unit
def test_propose_trades_suggests_correct_direction() -> None:
    targets = {"META": 0.5, "MSFT": 0.5}
    holdings = [_h("META", 60), _h("MSFT", 40)]
    prices = {"META": 100.0, "MSFT": 100.0}
    trades = rebalance.propose_trades(holdings, targets, prices, tolerance=0.05)
    by_sym = {t["symbol"]: t for t in trades}
    assert by_sym["META"]["side"] == "sell"
    assert by_sym["MSFT"]["side"] == "buy"
    # total_value=10000, target 50% each = 5000. META has 6000, need to sell 1000/100=10 shares
    assert by_sym["META"]["shares"] == pytest.approx(10.0, abs=0.1)
    assert by_sym["MSFT"]["shares"] == pytest.approx(10.0, abs=0.1)


@pytest.mark.unit
def test_propose_trades_empty_when_aligned() -> None:
    targets = {"META": 0.5, "MSFT": 0.5}
    holdings = [_h("META", 50), _h("MSFT", 50)]
    prices = {"META": 100.0, "MSFT": 100.0}
    assert rebalance.propose_trades(holdings, targets, prices, tolerance=0.05) == []
