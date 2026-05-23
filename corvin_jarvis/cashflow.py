"""Corvin Jarvis — Cash Flow Forecasting (Tier 3.3)

월별 지출 카테고리 ingest → 6개월 ahead forecasting → 자산 매도 권고 (advisory).

⚠️ 폐하가 expenses.json 작성 (각 카테고리 월별 평균). 변동 ingest는 차후 별도 모듈.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.cashflow")


def load_expenses(path: Path) -> dict[str, int]:
    """expenses.json → {category: monthly_krw}. 누락/오류 시 {}."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        cats = data.get("monthly_categories", {})
        return {str(k): int(v) for k, v in cats.items()}
    except (json.JSONDecodeError, OSError, TypeError) as e:
        log.warning("expenses load failed: %s", e)
        return {}


def monthly_total(expenses: dict[str, int]) -> int:
    """월 지출 총합."""
    return sum(int(v) for v in expenses.values())


def forecast(
    expenses: dict[str, int],
    current_cash_krw: int,
    months: int = 6,
) -> dict[str, Any]:
    """N개월 cash projection.

    가정: 지출은 일정, 수입 없음 (보수적). 음수 시점 = liquidity event 필요.
    """
    monthly = monthly_total(expenses)
    projections: list[dict[str, Any]] = []
    cash = int(current_cash_krw)
    months_until_negative: int | None = None
    for i in range(1, months + 1):
        cash_start = cash
        cash -= monthly
        projections.append({
            "month_idx": i,
            "cash_start_krw": cash_start,
            "monthly_outflow_krw": monthly,
            "cash_end_krw": cash,
        })
        if monthly > 0 and cash < 0 and months_until_negative is None:
            months_until_negative = i
    return {
        "starting_cash_krw": int(current_cash_krw),
        "monthly_outflow_krw": monthly,
        "projections": projections,
        "months_until_negative": months_until_negative,
    }
