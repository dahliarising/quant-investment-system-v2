"""Tests for run_signal_dryrun 순수 헬퍼 (I/O·라이브 fetch 제외)."""
from __future__ import annotations

import pytest

from corvin_jarvis import run_signal_dryrun as rsd
from corvin_jarvis import signal_router


@pytest.mark.unit
def test_holdings_from_portfolio_extracts_shares():
    pf = {"holdings": [
        {"symbol": "META", "shares": 7},
        {"symbol": "012450", "shares": 4},
        {"shares": 3},  # symbol 없음 → 무시
    ]}
    assert rsd.holdings_from_portfolio(pf) == {"META": 7, "012450": 4}


@pytest.mark.unit
def test_actionable_drops_hold_and_watch():
    verdicts = {
        "A": {"action": "매수"},
        "B": {"action": "홀딩"},
        "C": {"action": "관망"},
        "D": {"action": "비중축소"},
    }
    assert set(rsd.actionable(verdicts)) == {"A", "D"}


@pytest.mark.unit
def test_split_by_market_separates_kr_and_us():
    """국내(라우팅 가능) vs 해외(보류) 분리 — 통화/주문TR 다름."""
    candidates = {
        "012450": {"action": "비중축소"},  # KR
        "207940": {"action": "매도"},       # KR
        "AMD": {"action": "분할매수"},       # US
        "MSFT": {"action": "비중축소"},      # US
    }
    kr, us = rsd.split_by_market(candidates)
    assert set(kr) == {"012450", "207940"}
    assert set(us) == {"AMD", "MSFT"}


@pytest.mark.unit
def test_format_plans_empty():
    assert "없음" in rsd.format_plans([])


@pytest.mark.unit
def test_format_plans_renders_status_and_side():
    plans = [
        signal_router.RoutedPlan(
            intent=signal_router.OrderIntent("012450", "sell", 4, 1_228_000, "매도"),
            status="dry_run"),
    ]
    out = rsd.format_plans(plans)
    assert "012450" in out
    assert "sell" in out
    assert "매도" in out
    assert "dry_run" in out
