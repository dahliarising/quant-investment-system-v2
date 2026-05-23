"""Tests for corvin_jarvis.accounts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corvin_jarvis import accounts


@pytest.mark.unit
def test_load_accounts_returns_empty_when_missing(tmp_path: Path) -> None:
    out = accounts.load_accounts(tmp_path / "nope.json")
    assert out == []


@pytest.mark.unit
def test_load_accounts_parses_valid(tmp_path: Path) -> None:
    f = tmp_path / "accounts.json"
    f.write_text(json.dumps({
        "accounts": [
            {"id": "us-brokerage", "type": "us_stocks", "value_krw": 12000000, "liquidity": "T+2"},
            {"id": "kr-savings", "type": "krw_savings", "value_krw": 5000000, "liquidity": "T+0"},
            {"id": "real-estate", "type": "real_estate", "value_krw": 800000000, "liquidity": "months"},
        ]
    }))
    out = accounts.load_accounts(f)
    assert len(out) == 3


@pytest.mark.unit
def test_aggregate_by_type(tmp_path: Path) -> None:
    data = [
        {"id": "a", "type": "us_stocks", "value_krw": 10000000, "liquidity": "T+2"},
        {"id": "b", "type": "us_stocks", "value_krw": 2000000, "liquidity": "T+2"},
        {"id": "c", "type": "krw_savings", "value_krw": 5000000, "liquidity": "T+0"},
        {"id": "d", "type": "real_estate", "value_krw": 800000000, "liquidity": "months"},
    ]
    out = accounts.aggregate_by_type(data)
    assert out["us_stocks"] == 12000000
    assert out["krw_savings"] == 5000000
    assert out["real_estate"] == 800000000


@pytest.mark.unit
def test_aggregate_by_liquidity_orders_groups() -> None:
    data = [
        {"id": "a", "type": "us_stocks", "value_krw": 10000000, "liquidity": "T+2"},
        {"id": "b", "type": "real_estate", "value_krw": 800000000, "liquidity": "months"},
        {"id": "c", "type": "krw_savings", "value_krw": 5000000, "liquidity": "T+0"},
    ]
    out = accounts.aggregate_by_liquidity(data)
    assert list(out.keys()) == ["T+0", "T+2", "months"]


@pytest.mark.unit
def test_total_net_worth_sums_all() -> None:
    data = [
        {"id": "a", "type": "x", "value_krw": 1_000_000},
        {"id": "b", "type": "y", "value_krw": 2_000_000},
    ]
    assert accounts.total_net_worth(data) == 3_000_000
