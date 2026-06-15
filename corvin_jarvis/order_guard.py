"""Corvin Jarvis — 주문 안전장치 레이어 (Phase ②)

place_kr_order(kis_order)를 감싸는 정책 게이트. 주문 로직은 안 건드리고
"쏠지 말지"만 결정 — 관심사 분리.

3대 안전장치:
  1. 멱등성(idempotency) — 같은 신호/주문을 하루에 두 번 안 쏨 (영속 원장)
  2. 일일 상한 — 횟수(max_orders) + 금액(max_krw) 캡
  3. 🛡️ 실전 이중확인 — env=prod는 confirm_live=True *그리고*
     환경변수 CORVIN_ALLOW_LIVE_ORDERS=1 둘 다 있어야 통과 (defense in depth)

원장: state/order_ledger.json (날짜 바뀌면 자동 리셋). kis_token.json과 같은 패턴.

GuardedResult.status:
  "placed"    — 정상 주문됨
  "duplicate" — 멱등성 차단 (이미 오늘 같은 키)
  "blocked"   — 상한/실전가드 차단
  "rejected"  — 주문은 나갔으나 KIS가 거부 (원장 미기록 → 재시도 가능)
  "error"     — 입력/내부 오류
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from corvin_jarvis import kis_auth, kis_order

log = logging.getLogger("corvin.order.guard")

KST = ZoneInfo("Asia/Seoul")
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LEDGER = BASE_DIR / "state" / "order_ledger.json"

# 기본 일일 상한 (환경변수로 오버라이드)
DEFAULT_MAX_ORDERS = int(os.environ.get("CORVIN_MAX_ORDERS_PER_DAY", "10"))
DEFAULT_MAX_KRW = float(os.environ.get("CORVIN_MAX_KRW_PER_DAY", "5000000"))

LIVE_ENV_FLAG = "CORVIN_ALLOW_LIVE_ORDERS"

PlaceFn = Callable[..., kis_order.OrderResult]


@dataclass(frozen=True)
class GuardedResult:
    status: str  # placed | duplicate | blocked | rejected | error
    order_no: str | None = None
    error: str = ""
    key: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "placed"


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _load_ledger(path: Path, today: str) -> dict[str, Any]:
    """원장 로드. 날짜가 today와 다르면 새 날 → 빈 원장 반환 (리셋)."""
    if not path.exists():
        return {"date": today, "orders": []}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"date": today, "orders": []}
    if data.get("date") != today:
        return {"date": today, "orders": []}
    data.setdefault("orders", [])
    return data


def _save_ledger(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _make_key(
    symbol: str, side: str, qty: int, price: float | int | None,
    today: str, client_order_id: str | None,
) -> str:
    if client_order_id:
        return f"{today}:{client_order_id}"
    return f"{today}:{symbol}:{side}:{qty}:{price}"


def _live_allowed(env: kis_auth.KISEnv, confirm_live: bool) -> bool:
    """실전 주문 허용 조건 — confirm_live 플래그 + 환경변수 둘 다."""
    if env.env != "prod":
        return True  # 모의는 항상 통과 (kis_order가 자체 가드)
    return confirm_live and os.environ.get(LIVE_ENV_FLAG) == "1"


def place_order_safe(
    symbol: str,
    *,
    side: kis_order.Side,
    qty: int,
    price: float | int | None,
    order_type: kis_order.OrderType = "limit",
    env: kis_auth.KISEnv | None = None,
    confirm_live: bool = False,
    client_order_id: str | None = None,
    max_orders: int | None = None,
    max_krw: float | None = None,
    today: str | None = None,
    ledger_path: Path | None = None,
    place_fn: PlaceFn | None = None,
) -> GuardedResult:
    """안전장치를 통과한 주문만 place_fn으로 위임.

    Args:
        client_order_id: 멱등성 키 override (신호 ID 등). 없으면 파라미터로 자동 생성.
        max_orders / max_krw: 일일 상한 override (기본 환경변수/상수).
        today / ledger_path / place_fn: 테스트 주입용 (실사용 시 기본값).
    """
    env = env or kis_auth.load_env()
    today = today or _today_kst()
    ledger_path = ledger_path or DEFAULT_LEDGER
    place_fn = place_fn or kis_order.place_kr_order
    max_orders = DEFAULT_MAX_ORDERS if max_orders is None else max_orders
    max_krw = DEFAULT_MAX_KRW if max_krw is None else max_krw

    # 🛡️ 1. 실전 이중확인 — 가장 먼저, 상태 건드리기 전에
    if not _live_allowed(env, confirm_live):
        return GuardedResult(
            status="blocked",
            error=f"실전 주문 차단 — confirm_live + {LIVE_ENV_FLAG}=1 둘 다 필요",
        )

    # 2. 입력 sanity (상한 회계용 최소 검증, 나머지는 place_fn이 검증)
    if qty <= 0:
        return GuardedResult(status="error", error=f"수량은 양수여야 함: {qty}")

    key = _make_key(symbol, side, qty, price, today, client_order_id)
    ledger = _load_ledger(ledger_path, today)
    orders = ledger["orders"]

    # 3. 멱등성 — 오늘 같은 키 있으면 중복
    if any(o.get("key") == key for o in orders):
        log.info("중복 주문 차단 key=%s", key)
        return GuardedResult(status="duplicate", error="이미 오늘 동일 주문 존재", key=key)

    # 4. 일일 횟수 상한
    if len(orders) >= max_orders:
        return GuardedResult(
            status="blocked", key=key,
            error=f"일일 주문 횟수 상한 도달 ({len(orders)}/{max_orders})",
        )

    # 5. 일일 금액 상한 (시장가 price=None은 0으로 회계 → 횟수캡으로만 방어)
    est_krw = qty * (price or 0)
    spent = sum(o.get("krw", 0) for o in orders)
    if est_krw and spent + est_krw > max_krw:
        return GuardedResult(
            status="blocked", key=key,
            error=f"일일 주문 금액 상한 초과 ({spent + est_krw:,.0f} > {max_krw:,.0f}원)",
        )

    # 6. 위임 실행
    result = place_fn(
        symbol, side=side, qty=qty, price=price,
        order_type=order_type, env=env, confirm_live=confirm_live,
    )

    # 7. 성공만 원장 기록 (거부는 재시도 가능하게 미기록)
    if not result.ok:
        log.info("주문 거부 — 원장 미기록 key=%s err=%s", key, result.error)
        return GuardedResult(status="rejected", error=result.error, key=key, raw=result.raw)

    orders.append({
        "key": key,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "krw": est_krw,
        "order_no": result.order_no,
        "ts": datetime.now(KST).isoformat(timespec="seconds"),
    })
    _save_ledger(ledger_path, ledger)
    log.info("주문 기록 key=%s ODNO=%s (오늘 %d건)", key, result.order_no, len(orders))
    return GuardedResult(status="placed", order_no=result.order_no, key=key, raw=result.raw)
