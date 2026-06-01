"""Tests for corvin_jarvis.financial_metrics (mock raw, no real API)."""
from __future__ import annotations

import pytest

from corvin_jarvis import financial_metrics as fm


@pytest.mark.unit
def test_parse_us_financials_full():
    info = {
        "earningsGrowth": 0.62,
        "revenueGrowth": 0.33,
        "debtToEquity": 35.6,       # percent form
        "operatingMargins": 0.40,
    }
    m = fm.parse_us_financials(info)
    assert abs(m["eps_growth_yoy"] - 0.62) < 1e-9
    assert abs(m["revenue_growth_yoy"] - 0.33) < 1e-9
    assert abs(m["debt_ratio"] - 0.356) < 1e-6   # 35.6/100


@pytest.mark.unit
def test_parse_us_financials_partial():
    m = fm.parse_us_financials({"earningsGrowth": 0.1})
    assert m == {"eps_growth_yoy": 0.1}


@pytest.mark.unit
def test_parse_us_financials_empty():
    assert fm.parse_us_financials({}) == {}


@pytest.mark.unit
def test_parse_us_financials_ignores_none():
    m = fm.parse_us_financials({"earningsGrowth": None, "revenueGrowth": 0.2})
    assert "eps_growth_yoy" not in m
    assert m["revenue_growth_yoy"] == 0.2


@pytest.mark.unit
def test_parse_kr_fundamental_growth():
    # EPS 1년전 1000 → 현재 1300 = +30%
    m = fm.parse_kr_fundamental(eps_now=1300.0, eps_year_ago=1000.0)
    assert abs(m["eps_growth_yoy"] - 0.30) < 1e-9


@pytest.mark.unit
def test_parse_kr_fundamental_negative_eps_ago():
    # 흑자전환(이전 적자) → 성장률 정의 불가 → 빈 dict
    assert fm.parse_kr_fundamental(eps_now=500.0, eps_year_ago=-100.0) == {}


@pytest.mark.unit
def test_parse_kr_fundamental_missing():
    assert fm.parse_kr_fundamental(eps_now=None, eps_year_ago=1000.0) == {}
    assert fm.parse_kr_fundamental(eps_now=1300.0, eps_year_ago=None) == {}
