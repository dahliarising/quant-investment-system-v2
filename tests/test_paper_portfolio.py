"""Tests for corvin_jarvis.paper_portfolio — 현금관리 가상 포트폴리오 (매수+매도).

전략 무관 회계 엔진. 종목선정·가드·사이징은 별도 레이어.
모든 가격은 KRW 환산 기준(미국 체결은 호출자가 FX 환산해 전달).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis import paper_portfolio as pp


def _pf(cash=10_000_000):
    return pp.PaperPortfolio(cash_krw=cash, initial_krw=cash)


# ============================================================
# 매수 (진입)
# ============================================================

@pytest.mark.unit
def test_buy_reduces_cash_and_adds_holding():
    f = pp.buy(_pf(), "005930", 2, 300_000)
    assert f.cash_krw == 10_000_000 - 600_000
    assert f.holdings["005930"].qty == 2
    assert f.holdings["005930"].avg_price_krw == 300_000


@pytest.mark.unit
def test_buy_weighted_average():
    f = pp.buy(_pf(), "005930", 2, 300_000)
    f = pp.buy(f, "005930", 2, 340_000)
    assert f.holdings["005930"].qty == 4
    assert f.holdings["005930"].avg_price_krw == 320_000


@pytest.mark.unit
def test_buy_insufficient_cash_raises():
    with pytest.raises(ValueError):
        pp.buy(_pf(cash=100_000), "005930", 1, 300_000)


@pytest.mark.unit
def test_buy_tags_currency():
    f = pp.buy(_pf(), "AAPL", 1, 250_000, currency="USD")
    assert f.holdings["AAPL"].currency == "USD"


# ============================================================
# 매도 (청산)
# ============================================================

@pytest.mark.unit
def test_sell_increases_cash_and_reduces_holding():
    f = pp.buy(_pf(), "005930", 3, 300_000)
    f = pp.sell(f, "005930", 1, 350_000)
    assert f.holdings["005930"].qty == 2
    assert f.cash_krw == 10_000_000 - 900_000 + 350_000


@pytest.mark.unit
def test_sell_realizes_pnl_and_removes_when_flat():
    f = pp.buy(_pf(), "005930", 2, 300_000)
    f = pp.sell(f, "005930", 2, 350_000)
    assert f.realized_pnl_krw == 100_000
    assert "005930" not in f.holdings


@pytest.mark.unit
def test_sell_more_than_held_raises():
    f = pp.buy(_pf(), "005930", 1, 300_000)
    with pytest.raises(ValueError):
        pp.sell(f, "005930", 5, 350_000)


# ============================================================
# Mark-to-market + 비중 (리스크 가드용)
# ============================================================

@pytest.mark.unit
def test_mark_to_market_pnl():
    f = pp.buy(_pf(), "005930", 2, 300_000)
    snap = pp.mark_to_market(f, {"005930": 350_000})
    assert snap["holdings_value_krw"] == 700_000
    assert snap["total_value_krw"] == 10_100_000
    assert snap["pnl_krw"] == 100_000
    assert round(snap["pnl_pct"], 2) == 1.0


@pytest.mark.unit
def test_mark_to_market_missing_price_uses_cost():
    f = pp.buy(_pf(), "005930", 1, 300_000)
    snap = pp.mark_to_market(f, {})
    assert snap["total_value_krw"] == 10_000_000  # 손익 0 (취득원가 평가)


@pytest.mark.unit
def test_position_weight_pct():
    f = pp.buy(_pf(), "005930", 10, 300_000)  # 3M of 10M total
    w = pp.position_weight(f, "005930", {"005930": 300_000})
    assert round(w, 1) == 30.0  # 3M / 10M


# ============================================================
# 영속화
# ============================================================

@pytest.mark.unit
def test_save_load_roundtrip(tmp_path: Path):
    f = pp.buy(_pf(), "005930", 2, 300_000, ts="t1", reason="진입")
    p = tmp_path / "pf.json"
    pp.save(f, p)
    g = pp.load(p, initial_krw=10_000_000)
    assert g.cash_krw == f.cash_krw
    assert g.holdings["005930"].qty == 2
    assert len(g.trades) == 1


@pytest.mark.unit
def test_load_missing_returns_fresh(tmp_path: Path):
    g = pp.load(tmp_path / "none.json", initial_krw=5_000_000)
    assert g.cash_krw == 5_000_000
    assert g.holdings == {}
