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
def test_fetch_watchlist_uses_yf_for_us_and_kr_for_korean() -> None:
    with patch.object(pulse, "_yf_quote") as mock_yf, \
         patch("corvin_jarvis.pulse.get_kr_stock_data") as mock_kr:
        mock_yf.return_value = pulse.Quote(price=350.0, pct_change=2.5, source="yfinance", error=None)
        mock_kr.return_value = {"현재가": 70000.0, "전일대비(%)": -1.2}

        result = pulse.fetch_watchlist(["TSLA", "005930"])

        assert len(result) == 2
        by_sym = {r["symbol"]: r for r in result}
        assert by_sym["TSLA"]["price"] == 350.0
        assert by_sym["TSLA"]["source"] == "yfinance"
        assert by_sym["005930"]["price"] == 70000.0
        assert by_sym["005930"]["source"] == "FinanceDataReader"


@pytest.mark.unit
def test_fetch_watchlist_empty_input_returns_empty() -> None:
    assert pulse.fetch_watchlist([]) == []
