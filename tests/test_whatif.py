"""Tests for corvin_jarvis.whatif simulator."""
from __future__ import annotations

import pytest

from corvin_jarvis import whatif


@pytest.mark.unit
def test_parse_trade_buy_at_market() -> None:
    t = whatif.parse_trade("buy META 10")
    assert t == whatif.Trade(side="buy", symbol="META", shares=10.0, price=None)


@pytest.mark.unit
def test_parse_trade_sell_with_price() -> None:
    t = whatif.parse_trade("sell 005930 5 @ 70000")
    assert t == whatif.Trade(side="sell", symbol="005930", shares=5.0, price=70000.0)


@pytest.mark.unit
def test_parse_trade_is_case_insensitive_side() -> None:
    t = whatif.parse_trade("BUY tsla 2.5 @ 350")
    assert t == whatif.Trade(side="buy", symbol="TSLA", shares=2.5, price=350.0)


@pytest.mark.unit
def test_parse_trade_rejects_invalid_side() -> None:
    with pytest.raises(ValueError, match="side"):
        whatif.parse_trade("hodl META 10")


@pytest.mark.unit
def test_parse_trade_rejects_non_positive_shares() -> None:
    with pytest.raises(ValueError, match="shares"):
        whatif.parse_trade("buy META 0")
    with pytest.raises(ValueError, match="shares"):
        whatif.parse_trade("sell META -5")


@pytest.mark.unit
def test_parse_trade_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        whatif.parse_trade("buy META")
    with pytest.raises(ValueError):
        whatif.parse_trade("")
