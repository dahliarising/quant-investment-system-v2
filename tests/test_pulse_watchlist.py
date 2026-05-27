"""Tests for pulse.py watchlist support."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from corvin_jarvis import pulse


@pytest.mark.unit
def test_load_watchlist_returns_symbols_from_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"watchlist": ["TSLA", "005930"]}))
    result = pulse.load_watchlist(config)
    assert result == ["TSLA", "005930"]


@pytest.mark.unit
def test_load_watchlist_returns_empty_when_missing(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"other_key": "value"}))
    result = pulse.load_watchlist(config)
    assert result == []


@pytest.mark.unit
def test_load_watchlist_returns_empty_when_file_missing(tmp_path: Path) -> None:
    config = tmp_path / "nonexistent.json"
    result = pulse.load_watchlist(config)
    assert result == []


@pytest.mark.unit
def test_fetch_watchlist_uses_quote_provider_for_each_symbol() -> None:
    from corvin_jarvis import quote_provider

    quotes = {
        "TSLA": quote_provider.Quote(price=350.0, pct_change=2.5, source="yfinance", error=None),
        "005930": quote_provider.Quote(price=70000.0, pct_change=-1.2, source="pykrx", error=None),
    }
    with patch.object(pulse.quote_provider, "get_stock_quote", side_effect=lambda s: quotes[s]):
        result = pulse.fetch_watchlist(["TSLA", "005930"])

    assert len(result) == 2
    by_sym = {r["symbol"]: r for r in result}
    assert by_sym["TSLA"]["price"] == 350.0
    assert by_sym["TSLA"]["source"] == "yfinance"
    assert by_sym["TSLA"]["pct_change"] == 2.5
    assert by_sym["005930"]["price"] == 70000.0
    assert by_sym["005930"]["source"] == "pykrx"


@pytest.mark.unit
def test_fetch_watchlist_empty_input_returns_empty() -> None:
    assert pulse.fetch_watchlist([]) == []
