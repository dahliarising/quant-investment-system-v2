"""Tests for corvin_jarvis.tax — KR tax-aware strategy."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from corvin_jarvis import tax


@pytest.mark.unit
def test_kr_gain_threshold_calc_within_limit() -> None:
    """한국 양도세: 250만원 한도 이하 비과세 (해외주식)."""
    out = tax.kr_capital_gain_taxable(realized_krw=2_000_000)
    assert out["taxable_krw"] == 0
    assert out["limit_krw"] == 2_500_000
    assert out["remaining_limit_krw"] == 500_000


@pytest.mark.unit
def test_kr_gain_above_limit_is_taxable() -> None:
    """5,000,000원 실현이익 → 2,500,000원 과세."""
    out = tax.kr_capital_gain_taxable(realized_krw=5_000_000)
    assert out["taxable_krw"] == 2_500_000
    assert out["remaining_limit_krw"] == 0


@pytest.mark.unit
def test_long_term_holdings_flagged() -> None:
    today = date(2026, 5, 23)
    holdings_with_dates = [
        {"symbol": "META", "shares": 7, "purchase_date": "2023-01-15"},  # >3y
        {"symbol": "MSFT", "shares": 6, "purchase_date": "2025-03-01"},  # <3y
    ]
    out = tax.long_term_holdings(holdings_with_dates, today=today, threshold_years=3)
    by_sym = {h["symbol"]: h for h in out}
    assert by_sym["META"]["is_long_term"] is True
    assert by_sym["META"]["years_held"] >= 3.0
    assert by_sym["MSFT"]["is_long_term"] is False


@pytest.mark.unit
def test_long_term_holdings_handles_missing_date() -> None:
    holdings = [{"symbol": "X", "shares": 1, "purchase_date": None}]
    out = tax.long_term_holdings(holdings, today=date(2026, 1, 1))
    assert out[0]["is_long_term"] is False
    assert out[0]["years_held"] is None


@pytest.mark.unit
def test_tax_loss_harvesting_candidates() -> None:
    """미실현 손실 종목 → harvesting 후보."""
    positions = [
        {"symbol": "META", "shares": 7, "avg_price": 597.61, "current_price": 605.0},  # 이익
        {"symbol": "PLTR", "shares": 100, "avg_price": 50.0, "current_price": 30.0},   # 큰 손실
        {"symbol": "AMD", "shares": 20, "avg_price": 120.0, "current_price": 110.0},   # 작은 손실
    ]
    candidates = tax.harvesting_candidates(positions, min_loss_pct=-5.0)
    syms = [c["symbol"] for c in candidates]
    assert "PLTR" in syms
    assert "AMD" in syms
    assert "META" not in syms
    # PLTR loss should be larger
    pltr = next(c for c in candidates if c["symbol"] == "PLTR")
    assert pltr["unrealized_loss_pct"] < -30


@pytest.mark.unit
def test_tax_loss_harvesting_threshold_filters() -> None:
    positions = [
        {"symbol": "AMD", "shares": 20, "avg_price": 120.0, "current_price": 117.0},  # -2.5%
    ]
    candidates = tax.harvesting_candidates(positions, min_loss_pct=-5.0)
    assert candidates == []
