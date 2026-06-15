"""Tests for corvin_jarvis.signal_router — verdict 신호 → 주문 intent 매핑 + 라우팅.

매핑(순수 함수)은 KIS 없이 검증. 실행은 guard를 주입해 검증.
"""
from __future__ import annotations

import pytest

from corvin_jarvis import kis_auth, signal_router


@pytest.fixture
def mock_env() -> kis_auth.KISEnv:
    return kis_auth.KISEnv(
        app_key="k", app_secret="s",
        base_url=kis_auth.MOCK_URL, env="mock", account_no="12345678-01",
    )


# ============================================================
# 매핑 — action → side/qty
# ============================================================

@pytest.mark.unit
def test_buy_sizes_by_krw_tranche():
    it = signal_router.intent_from_verdict(
        "005930", "매수", price=100_000, shares=0, buy_krw=500_000)
    assert it is not None
    assert it.side == "buy"
    assert it.qty == 5  # 500k / 100k


@pytest.mark.unit
def test_split_buy_uses_half_tranche():
    it = signal_router.intent_from_verdict(
        "005930", "분할매수", price=100_000, shares=0, buy_krw=500_000)
    assert it.side == "buy"
    assert it.qty == 2  # (500k*0.5)//100k


@pytest.mark.unit
def test_full_sell_uses_all_holdings():
    it = signal_router.intent_from_verdict(
        "005930", "매도", price=70_000, shares=7)
    assert it.side == "sell"
    assert it.qty == 7


@pytest.mark.unit
def test_partial_reduce_uses_fraction():
    it = signal_router.intent_from_verdict(
        "012450", "비중축소", price=1_000_000, shares=9, partial_fraction=1 / 3)
    assert it.side == "sell"
    assert it.qty == 3


@pytest.mark.unit
@pytest.mark.parametrize("action", ["홀딩", "관망"])
def test_hold_and_watch_produce_no_intent(action):
    assert signal_router.intent_from_verdict(
        "005930", action, price=100, shares=5) is None


@pytest.mark.unit
def test_sell_skipped_when_not_held():
    """매도 신호인데 보유 0 → intent 없음 (없는 걸 못 팜)."""
    assert signal_router.intent_from_verdict(
        "005930", "매도", price=100, shares=0) is None


@pytest.mark.unit
def test_buy_skipped_when_no_price():
    assert signal_router.intent_from_verdict(
        "005930", "매수", price=None, shares=0, buy_krw=500_000) is None


@pytest.mark.unit
def test_buy_skipped_when_tranche_too_small_for_one_share():
    """트랜치로 1주도 못 사면 intent 없음."""
    assert signal_router.intent_from_verdict(
        "005930", "매수", price=600_000, shares=0, buy_krw=500_000) is None


# ============================================================
# 라우팅 — dry-run vs 실행
# ============================================================

@pytest.mark.unit
def test_route_dry_run_places_nothing(mock_env):
    verdicts = {
        "005930": {"action": "매수", "name": "삼성전자"},
        "000660": {"action": "관망", "name": "SK하이닉스"},
    }
    prices = {"005930": 100_000}
    calls: list = []
    def safe_fn(*a, **k):
        calls.append(k)
        raise AssertionError("dry-run인데 주문 호출됨")
    plans = signal_router.route_verdicts(
        verdicts, holdings={}, prices=prices, env=mock_env,
        place=False, buy_krw=500_000, safe_fn=safe_fn,
    )
    # 매수 1건만 intent, 관망은 제외
    assert len(plans) == 1
    assert plans[0].intent.symbol == "005930"
    assert plans[0].status == "dry_run"
    assert calls == []  # 한 발도 안 나감


@pytest.mark.unit
def test_route_place_routes_through_guard(mock_env):
    from corvin_jarvis import order_guard
    verdicts = {"005930": {"action": "매수", "name": "삼성전자"}}
    prices = {"005930": 100_000}
    seen: list = []
    def safe_fn(symbol, **k):
        seen.append((symbol, k.get("side"), k.get("qty"), k.get("client_order_id")))
        return order_guard.GuardedResult(status="placed", order_no="ODNO1")
    plans = signal_router.route_verdicts(
        verdicts, holdings={}, prices=prices, env=mock_env,
        place=True, buy_krw=500_000, today="2026-06-15", safe_fn=safe_fn,
    )
    assert plans[0].status == "placed"
    assert seen[0][0] == "005930"
    assert seen[0][1] == "buy"
    # client_order_id는 action 기반 → 같은 날 재실행 시 guard가 중복으로 막음
    assert "매수" in seen[0][3]
