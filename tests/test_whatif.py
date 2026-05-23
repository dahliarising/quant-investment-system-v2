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


# ---- apply_trade ----


def _holding(symbol: str, shares: float, avg_usd: float = 100.0, currency: str = "USD") -> dict:
    return {
        "symbol": symbol, "shares": shares,
        "avgPriceUSD": avg_usd, "avgPriceKRW": avg_usd * 1500,
        "currency": currency,
    }


@pytest.mark.unit
def test_apply_buy_creates_new_position() -> None:
    holdings = [_holding("META", 7, 597.61, "USD")]
    trade = whatif.Trade(side="buy", symbol="TSLA", shares=3, price=350.0)
    out = whatif.apply_trade(holdings, trade)
    syms = {h["symbol"]: h for h in out}
    assert "TSLA" in syms
    assert syms["TSLA"]["shares"] == 3
    assert syms["TSLA"]["avgPriceUSD"] == 350.0
    assert syms["META"]["shares"] == 7  # 기존 변경 없음


@pytest.mark.unit
def test_apply_buy_averages_existing_position() -> None:
    """META 7주 @ 597.61 + 3주 @ 620 → 10주 @ (7*597.61+3*620)/10 = 604.34"""
    holdings = [_holding("META", 7, 597.61, "USD")]
    trade = whatif.Trade(side="buy", symbol="META", shares=3, price=620.0)
    out = whatif.apply_trade(holdings, trade)
    assert len(out) == 1
    assert out[0]["shares"] == 10
    expected = (7 * 597.61 + 3 * 620) / 10
    assert out[0]["avgPriceUSD"] == pytest.approx(expected, abs=0.01)


@pytest.mark.unit
def test_apply_sell_reduces_shares() -> None:
    holdings = [_holding("META", 7, 597.61, "USD")]
    trade = whatif.Trade(side="sell", symbol="META", shares=3, price=620.0)
    out = whatif.apply_trade(holdings, trade)
    assert out[0]["shares"] == 4
    assert out[0]["avgPriceUSD"] == 597.61  # avg 유지


@pytest.mark.unit
def test_apply_sell_full_position_removes() -> None:
    holdings = [_holding("META", 7, 597.61, "USD"), _holding("MSFT", 6, 350.0)]
    trade = whatif.Trade(side="sell", symbol="META", shares=7, price=620.0)
    out = whatif.apply_trade(holdings, trade)
    assert len(out) == 1
    assert out[0]["symbol"] == "MSFT"


@pytest.mark.unit
def test_apply_sell_more_than_owned_raises() -> None:
    holdings = [_holding("META", 7, 597.61)]
    trade = whatif.Trade(side="sell", symbol="META", shares=10, price=620.0)
    with pytest.raises(ValueError, match="exceeds"):
        whatif.apply_trade(holdings, trade)


@pytest.mark.unit
def test_apply_sell_unknown_symbol_raises() -> None:
    holdings = [_holding("META", 7)]
    trade = whatif.Trade(side="sell", symbol="TSLA", shares=1, price=350.0)
    with pytest.raises(ValueError, match="not held"):
        whatif.apply_trade(holdings, trade)


@pytest.mark.unit
def test_apply_buy_at_market_requires_market_price() -> None:
    """price=None일 때 market_prices에서 lookup 필수."""
    holdings = []
    trade = whatif.Trade(side="buy", symbol="META", shares=3, price=None)
    with pytest.raises(ValueError, match="market price"):
        whatif.apply_trade(holdings, trade, market_prices={})

    out = whatif.apply_trade(holdings, trade, market_prices={"META": 605.06})
    assert out[0]["avgPriceUSD"] == 605.06
