"""Tests for corvin_jarvis.order_guard — 주문 안전장치 레이어.

place_kr_order를 감싸는 정책: 멱등성 + 일일상한 + 실전 이중확인.
place_fn을 주입해 실제 KIS 호출 없이 정책만 검증.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis import kis_auth, kis_order, order_guard


@pytest.fixture
def mock_env() -> kis_auth.KISEnv:
    return kis_auth.KISEnv(
        app_key="k", app_secret="s",
        base_url=kis_auth.MOCK_URL, env="mock", account_no="12345678-01",
    )


@pytest.fixture
def prod_env() -> kis_auth.KISEnv:
    return kis_auth.KISEnv(
        app_key="k", app_secret="s",
        base_url=kis_auth.PROD_URL, env="prod", account_no="12345678-01",
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "order_ledger.json"


def _ok_fn(calls: list[dict]):
    """성공하는 place_fn 스텁 — 호출 기록."""
    def fn(symbol, *, side, qty, price, order_type="limit", env=None, confirm_live=False):
        calls.append({"symbol": symbol, "side": side, "qty": qty,
                      "confirm_live": confirm_live})
        return kis_order.OrderResult(ok=True, order_no="ODNO%d" % len(calls))
    return fn


# ============================================================
# 멱등성 — 중복 주문 차단
# ============================================================

@pytest.mark.unit
def test_duplicate_same_params_blocked(mock_env, ledger):
    calls: list[dict] = []
    fn = _ok_fn(calls)
    args = dict(side="buy", qty=3, price=70000, env=mock_env,
                today="2026-06-15", ledger_path=ledger, place_fn=fn)
    r1 = order_guard.place_order_safe("005930", **args)
    r2 = order_guard.place_order_safe("005930", **args)
    assert r1.status == "placed"
    assert r2.status == "duplicate"
    assert len(calls) == 1  # 두 번째는 실제 주문 안 나감


@pytest.mark.unit
def test_distinct_orders_both_placed(mock_env, ledger):
    calls: list[dict] = []
    fn = _ok_fn(calls)
    base = dict(side="buy", qty=1, price=100, env=mock_env,
                today="2026-06-15", ledger_path=ledger, place_fn=fn)
    order_guard.place_order_safe("005930", **base)
    r2 = order_guard.place_order_safe("000660", **base)
    assert r2.status == "placed"
    assert len(calls) == 2


@pytest.mark.unit
def test_explicit_client_order_id_dedupes(mock_env, ledger):
    calls: list[dict] = []
    fn = _ok_fn(calls)
    base = dict(env=mock_env, today="2026-06-15", ledger_path=ledger, place_fn=fn)
    order_guard.place_order_safe("005930", side="buy", qty=1, price=100,
                                 client_order_id="sig-42", **base)
    r2 = order_guard.place_order_safe("005930", side="buy", qty=9, price=999,
                                      client_order_id="sig-42", **base)
    assert r2.status == "duplicate"  # 파라미터 달라도 같은 id면 중복
    assert len(calls) == 1


# ============================================================
# 일일 상한 — 횟수 / 금액
# ============================================================

@pytest.mark.unit
def test_daily_count_cap_blocks(mock_env, ledger):
    calls: list[dict] = []
    fn = _ok_fn(calls)
    base = dict(side="buy", price=100, env=mock_env, today="2026-06-15",
                ledger_path=ledger, place_fn=fn, max_orders=2)
    order_guard.place_order_safe("AAA", qty=1, **base)
    order_guard.place_order_safe("BBB", qty=1, **base)
    r3 = order_guard.place_order_safe("CCC", qty=1, **base)
    assert r3.status == "blocked"
    assert "횟수" in r3.error or "count" in r3.error.lower()
    assert len(calls) == 2


@pytest.mark.unit
def test_daily_krw_cap_blocks(mock_env, ledger):
    calls: list[dict] = []
    fn = _ok_fn(calls)
    # cap 1,000,000원. 첫 주문 700k OK, 두번째 500k → 합 1.2M 초과 차단
    base = dict(side="buy", env=mock_env, today="2026-06-15",
                ledger_path=ledger, place_fn=fn, max_krw=1_000_000)
    r1 = order_guard.place_order_safe("AAA", qty=1, price=700_000, **base)
    r2 = order_guard.place_order_safe("BBB", qty=1, price=500_000, **base)
    assert r1.status == "placed"
    assert r2.status == "blocked"
    assert len(calls) == 1


# ============================================================
# 원장 — 날짜 바뀌면 카운터 리셋
# ============================================================

@pytest.mark.unit
def test_ledger_resets_on_new_day(mock_env, ledger):
    calls: list[dict] = []
    fn = _ok_fn(calls)
    common = dict(side="buy", qty=1, price=100, env=mock_env,
                  ledger_path=ledger, place_fn=fn, max_orders=1)
    order_guard.place_order_safe("AAA", today="2026-06-15", **common)
    # 같은 날 두번째 → 막힘
    blocked = order_guard.place_order_safe("BBB", today="2026-06-15", **common)
    assert blocked.status == "blocked"
    # 다음 날 → 카운터 리셋, 통과
    nextday = order_guard.place_order_safe("CCC", today="2026-06-16", **common)
    assert nextday.status == "placed"


# ============================================================
# 🛡️ 실전 이중확인 (defense in depth)
# ============================================================

@pytest.mark.unit
def test_live_blocked_without_env_flag(prod_env, ledger, monkeypatch):
    """실전 + confirm_live=True 여도 CORVIN_ALLOW_LIVE_ORDERS 없으면 차단."""
    monkeypatch.delenv("CORVIN_ALLOW_LIVE_ORDERS", raising=False)
    calls: list[dict] = []
    fn = _ok_fn(calls)
    r = order_guard.place_order_safe(
        "005930", side="buy", qty=1, price=100, env=prod_env,
        confirm_live=True, today="2026-06-15", ledger_path=ledger, place_fn=fn,
    )
    assert r.status == "blocked"
    assert len(calls) == 0  # 실전 주문 한 발도 안 나감


@pytest.mark.unit
def test_live_allowed_with_both_gates(prod_env, ledger, monkeypatch):
    monkeypatch.setenv("CORVIN_ALLOW_LIVE_ORDERS", "1")
    calls: list[dict] = []
    fn = _ok_fn(calls)
    r = order_guard.place_order_safe(
        "005930", side="buy", qty=1, price=100, env=prod_env,
        confirm_live=True, today="2026-06-15", ledger_path=ledger, place_fn=fn,
    )
    assert r.status == "placed"
    assert calls[0]["confirm_live"] is True  # 실전 플래그 전달됨


# ============================================================
# 실패한 주문은 원장에 기록 안 함 (재시도 가능)
# ============================================================

@pytest.mark.unit
def test_rejected_order_not_recorded(mock_env, ledger):
    def reject_fn(symbol, *, side, qty, price, order_type="limit", env=None, confirm_live=False):
        return kis_order.OrderResult(ok=False, error="잔고부족")
    base = dict(side="buy", qty=1, price=100, env=mock_env,
                today="2026-06-15", ledger_path=ledger)
    r1 = order_guard.place_order_safe("005930", place_fn=reject_fn, **base)
    assert r1.status == "rejected"
    # 거부됐으니 같은 주문 재시도 가능 (중복 아님)
    calls: list[dict] = []
    r2 = order_guard.place_order_safe("005930", place_fn=_ok_fn(calls), **base)
    assert r2.status == "placed"
    assert len(calls) == 1
