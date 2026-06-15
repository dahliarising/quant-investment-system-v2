"""Corvin Jarvis — 신호→주문 라우터 (Phase ③)

verdict(signals/verdict.py) 액션을 주문 intent로 매핑하고, ②안전장치(order_guard)를
거쳐 모의주문까지 연결. 기본은 dry-run(주문 안 나감, "뭘 살지"만 보여줌).

매핑 정책 (보수적 기본값 — 폐하 조정 가능):
    매수 / 비중확대  → buy,  고정 트랜치(buy_krw)만큼
    분할매수          → buy,  트랜치의 절반(SPLIT_RATIO)
    비중축소          → sell, 보유분 × partial_fraction (기본 1/3)
    매도              → sell, 보유 전량
    홀딩 / 관망       → 주문 없음

매핑은 순수 함수(intent_from_verdict) — KIS·guard 없이 검증 가능.
실행(route_verdicts)은 dry-run 기본, place=True일 때만 guard로 위임.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from corvin_jarvis import kis_auth, order_guard

log = logging.getLogger("corvin.signal.router")

BUY_ACTIONS = frozenset({"매수", "비중확대"})
SPLIT_BUY_ACTIONS = frozenset({"분할매수"})
FULL_SELL_ACTIONS = frozenset({"매도"})
PARTIAL_SELL_ACTIONS = frozenset({"비중축소"})
# 홀딩·관망 등 그 외 = 주문 없음

SPLIT_RATIO = 0.5            # 분할매수 = 트랜치 절반
DEFAULT_BUY_KRW = 500_000   # 매수 1트랜치 기본 금액
DEFAULT_PARTIAL_FRACTION = 1 / 3  # 비중축소 = 보유 1/3


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str          # buy | sell
    qty: int
    price: float
    action: str        # 원본 verdict 액션
    reason: str = ""


@dataclass(frozen=True)
class RoutedPlan:
    intent: OrderIntent
    status: str        # dry_run | placed | duplicate | blocked | rejected | error
    order_no: str | None = None
    detail: str = ""


def intent_from_verdict(
    symbol: str,
    action: str,
    *,
    price: float | int | None,
    shares: int,
    buy_krw: float = DEFAULT_BUY_KRW,
    partial_fraction: float = DEFAULT_PARTIAL_FRACTION,
) -> OrderIntent | None:
    """verdict 액션 1건 → OrderIntent. 주문 불가/불필요 시 None.

    Args:
        price: 현재가(지정가 기준). 없으면 매수/매도 모두 None.
        shares: 현재 보유 수량(매도 사이징용).
    """
    if price in (None, 0):
        return None
    price_f = float(price)

    if action in BUY_ACTIONS or action in SPLIT_BUY_ACTIONS:
        tranche = buy_krw * (SPLIT_RATIO if action in SPLIT_BUY_ACTIONS else 1.0)
        qty = int(tranche // price_f)
        if qty < 1:
            return None  # 트랜치로 1주도 못 삼
        return OrderIntent(symbol, "buy", qty, price_f, action,
                           reason=f"{action} — 트랜치 {tranche:,.0f}원 / {price_f:,.0f}")

    if action in FULL_SELL_ACTIONS:
        if shares < 1:
            return None  # 없는 걸 못 팜
        return OrderIntent(symbol, "sell", int(shares), price_f, action,
                           reason="매도 — 보유 전량 청산")

    if action in PARTIAL_SELL_ACTIONS:
        qty = int(shares * partial_fraction)
        if qty < 1:
            return None
        return OrderIntent(symbol, "sell", qty, price_f, action,
                           reason=f"비중축소 — 보유 {shares}주 × {partial_fraction:.0%}")

    return None  # 홀딩·관망 등


SafeFn = Callable[..., order_guard.GuardedResult]


def route_verdicts(
    verdicts: dict[str, dict[str, Any]],
    *,
    holdings: dict[str, int],
    prices: dict[str, float],
    env: kis_auth.KISEnv | None = None,
    place: bool = False,
    buy_krw: float = DEFAULT_BUY_KRW,
    partial_fraction: float = DEFAULT_PARTIAL_FRACTION,
    confirm_live: bool = False,
    today: str | None = None,
    safe_fn: SafeFn | None = None,
) -> list[RoutedPlan]:
    """verdicts 전체를 intent로 변환 후 dry-run 또는 guard 실행.

    place=False(기본): 주문 안 나감, intent만 RoutedPlan(status="dry_run")으로 반환.
    place=True: order_guard.place_order_safe로 위임 (모의 기본, 실전은 이중확인).
    client_order_id = "{symbol}:{action}" → 같은 날 같은 신호 재실행은 guard가 중복 차단.
    """
    safe_fn = safe_fn or order_guard.place_order_safe
    plans: list[RoutedPlan] = []

    for symbol, v in verdicts.items():
        action = v.get("action", "")
        intent = intent_from_verdict(
            symbol, action,
            price=prices.get(symbol),
            shares=holdings.get(symbol, 0),
            buy_krw=buy_krw,
            partial_fraction=partial_fraction,
        )
        if intent is None:
            continue

        if not place:
            plans.append(RoutedPlan(intent=intent, status="dry_run",
                                    detail="dry-run — 주문 미실행"))
            continue

        result = safe_fn(
            intent.symbol, side=intent.side, qty=intent.qty, price=intent.price,
            env=env, confirm_live=confirm_live,
            client_order_id=f"{symbol}:{action}", today=today,
        )
        plans.append(RoutedPlan(
            intent=intent, status=result.status,
            order_no=result.order_no, detail=result.error,
        ))

    return plans
