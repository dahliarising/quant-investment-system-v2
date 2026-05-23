"""Tests for corvin_jarvis.cashflow."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corvin_jarvis import cashflow


@pytest.mark.unit
def test_load_expenses_returns_empty_when_missing(tmp_path: Path) -> None:
    assert cashflow.load_expenses(tmp_path / "nope.json") == {}


@pytest.mark.unit
def test_load_expenses_parses_categories(tmp_path: Path) -> None:
    f = tmp_path / "expenses.json"
    f.write_text(json.dumps({
        "monthly_categories": {
            "생활비": 3000000,
            "자녀": 1500000,
            "대출": 1200000,
            "세금": 500000,
        }
    }))
    out = cashflow.load_expenses(f)
    assert out["생활비"] == 3000000
    assert out["자녀"] == 1500000


@pytest.mark.unit
def test_monthly_total_sums_categories() -> None:
    expenses = {"생활비": 3000000, "자녀": 1500000, "대출": 1200000, "세금": 500000}
    assert cashflow.monthly_total(expenses) == 6200000


@pytest.mark.unit
def test_forecast_horizon_returns_n_months() -> None:
    expenses = {"생활비": 3000000, "자녀": 1500000}
    f = cashflow.forecast(expenses, current_cash_krw=20_000_000, months=6)
    assert len(f["projections"]) == 6
    # 매월 4.5M 지출, cash 20M → 4-5개월 후 마이너스
    cash_values = [p["cash_end_krw"] for p in f["projections"]]
    assert cash_values[0] > 0
    assert any(v < 0 for v in cash_values[3:])  # 4개월 이후 부족
    assert f["months_until_negative"] in {4, 5}


@pytest.mark.unit
def test_forecast_handles_zero_expense() -> None:
    f = cashflow.forecast({}, current_cash_krw=10_000_000, months=6)
    assert f["months_until_negative"] is None
    assert all(p["cash_end_krw"] == 10_000_000 for p in f["projections"])