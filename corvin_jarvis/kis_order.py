"""Corvin Jarvis — KIS 주문 모듈 (Tier 2.5, 모의 우선)

국내주식 현금주문(매수/매도) + 잔고조회. 시세(kis_quote)와 달리:
  - POST 메서드 + 주문 body
  - hashkey 헤더 필수 (POST /uapi/hashkey 선행)
  - 🛡️ 모의→실전 가드: env="prod"는 confirm_live=True 없이는 차단

⚠️ 안전 범위 (Phase ①): 본 모듈은 *모의계좌(mock)* 주문이 기본.
   실전 주문은 명시적 confirm_live=True를 줘야만 나감 (이중 방어).
   멱등성·일일상한 등 상위 안전장치는 Phase ②에서 별도 래퍼로 추가.

TR_ID (KIS 공식 스펙 기준 — 실전 전환 전 apiportal.koreainvestment.com에서 재확인):
    국내 매수:  실전 TTTC0802U / 모의 VTTC0802U
    국내 매도:  실전 TTTC0801U / 모의 VTTC0801U
    잔고 조회:  실전 TTTC8434R / 모의 VTTC8434R
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import requests

from corvin_jarvis import kis_auth

log = logging.getLogger("corvin.kis.order")

# 실전 TR_ID — 모의는 첫 글자 T→V 치환 (_resolve_tr_id)
TR_KR_BUY = "TTTC0802U"
TR_KR_SELL = "TTTC0801U"
TR_KR_BALANCE = "TTTC8434R"

ORDER_PATH = "/uapi/domestic-stock/v1/trading/order-cash"
BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"
HASHKEY_PATH = "/uapi/hashkey"

REQUEST_TIMEOUT = 10
RATE_LIMIT_DELAY = 0.06
MAX_RETRIES_ON_RATE_LIMIT = 3
RATE_LIMIT_BACKOFF_BASE = 0.8
_RATE_LIMIT_CODES = frozenset({"EGW00201", "EGW00301"})

# 주문구분 (ORD_DVSN)
DVSN_LIMIT = "00"   # 지정가
DVSN_MARKET = "01"  # 시장가

Side = Literal["buy", "sell"]
OrderType = Literal["limit", "market"]


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    order_no: str | None = None
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Balance:
    orderable_cash: float | None
    total_eval: float | None
    holdings: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _resolve_tr_id(prod_tr_id: str, env: kis_auth.KISEnv) -> str:
    """모의(mock)는 TR_ID 첫 글자 T→V. 실전은 그대로."""
    if env.env == "mock" and prod_tr_id.startswith("T"):
        return "V" + prod_tr_id[1:]
    return prod_tr_id


def _split_account(account_no: str) -> tuple[str, str]:
    """'12345678-01' → ('12345678', '01'). '-' 없으면 뒤 2자리를 상품코드로."""
    acct = (account_no or "").strip()
    if "-" in acct:
        cano, _, prdt = acct.partition("-")
        return cano.strip(), prdt.strip()
    digits = "".join(ch for ch in acct if ch.isdigit())
    if len(digits) > 2:
        return digits[:-2], digits[-2:]
    return digits, ""


def _is_rate_limit_error(data: dict[str, Any]) -> bool:
    return str(data.get("msg_cd") or "") in _RATE_LIMIT_CODES


def _hashkey(env: kis_auth.KISEnv, body: dict[str, str]) -> str | None:
    """POST /uapi/hashkey → 주문 body의 무결성 해시. 주문 헤더 필수값."""
    url = f"{env.base_url}{HASHKEY_PATH}"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "appkey": env.app_key,
        "appsecret": env.app_secret,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        log.warning("KIS hashkey 요청 실패: %s", exc)
        return None
    if r.status_code != 200:
        log.warning("KIS hashkey → %d: %s", r.status_code, r.text[:200])
        return None
    try:
        return r.json().get("HASH")
    except (ValueError, AttributeError):
        return None


def _http_post(
    env: kis_auth.KISEnv,
    path: str,
    tr_id: str,
    body: dict[str, str],
    *,
    hashkey: str | None = None,
) -> dict[str, Any]:
    """주문 POST — rate-limit 재시도 + backoff (kis_quote._http_get 미러)."""
    url = f"{env.base_url}{path}"
    headers = kis_auth.build_headers(env, tr_id)
    if hashkey:
        headers["hashkey"] = hashkey

    last_data: dict[str, Any] = {}
    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
        r = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
        try:
            data = r.json()
        except (ValueError, AttributeError):
            data = {}
        last_data = data

        if r.status_code in (429, 500) and _is_rate_limit_error(data):
            if attempt < MAX_RETRIES_ON_RATE_LIMIT:
                backoff = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                log.info("KIS order rate-limit (attempt %d) — sleep %.2fs", attempt + 1, backoff)
                time.sleep(backoff)
                continue
            log.warning("KIS %s rate-limit exhausted", tr_id)
            return last_data

        if r.status_code != 200:
            log.warning("KIS %s %s → %d: %s", tr_id, path, r.status_code, r.text[:200])
            return data or {"rt_cd": "1", "msg1": f"HTTP {r.status_code}"}

        time.sleep(RATE_LIMIT_DELAY)
        return data

    return last_data


def place_kr_order(
    symbol: str,
    *,
    side: Side,
    qty: int,
    price: float | int | None,
    order_type: OrderType = "limit",
    env: kis_auth.KISEnv | None = None,
    confirm_live: bool = False,
) -> OrderResult:
    """국내주식 현금주문. 기본 모의(mock). 실전은 confirm_live=True 필수.

    Args:
        symbol: 종목코드 (예: "005930")
        side: "buy" | "sell"
        qty: 주문 수량 (>0)
        price: 지정가. order_type="market"이면 무시(None 허용).
        order_type: "limit"(지정가) | "market"(시장가)
        confirm_live: 실전(prod) 주문 명시 승인. 모의에선 무시.
    """
    env = env or kis_auth.load_env()

    # 🛡️ 모의→실전 가드 — 약속: 실전 안 건드림
    if env.env != "mock" and not confirm_live:
        return OrderResult(
            ok=False,
            error="실전 주문 차단 — confirm_live=True 명시 필요 (현재 env=%s)" % env.env,
        )

    # 입력 검증
    if side not in ("buy", "sell"):
        return OrderResult(ok=False, error=f"알 수 없는 side: {side!r}")
    if qty <= 0:
        return OrderResult(ok=False, error=f"수량은 양수여야 함: {qty}")
    if order_type == "limit" and not price:
        return OrderResult(ok=False, error="지정가 주문은 price 필요")

    cano, prdt = _split_account(env.account_no)
    if not cano:
        return OrderResult(ok=False, error="KIS_ACCOUNT_NO 미설정 — 주문 불가")

    dvsn = DVSN_MARKET if order_type == "market" else DVSN_LIMIT
    unpr = "0" if order_type == "market" else str(int(price))  # type: ignore[arg-type]
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "PDNO": symbol,
        "ORD_DVSN": dvsn,
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": unpr,
    }

    prod_tr = TR_KR_BUY if side == "buy" else TR_KR_SELL
    tr_id = _resolve_tr_id(prod_tr, env)

    hashkey = _hashkey(env, body)
    if not hashkey:
        return OrderResult(ok=False, error="hashkey 발급 실패 — 주문 중단")

    data = _http_post(env, ORDER_PATH, tr_id, body, hashkey=hashkey)
    rt_cd = str(data.get("rt_cd", ""))
    if rt_cd != "0":
        return OrderResult(ok=False, error=data.get("msg1", "주문 거부") or "주문 거부", raw=data)

    order_no = (data.get("output") or {}).get("ODNO")
    log.info("KIS 주문 체결요청 OK env=%s %s %s x%d → ODNO=%s",
             env.env, side, symbol, qty, order_no)
    return OrderResult(ok=True, order_no=order_no, raw=data)


def get_kr_balance(env: kis_auth.KISEnv | None = None) -> Balance:
    """국내 잔고 + 주문가능현금. inquire-balance(TTTC8434R)."""
    env = env or kis_auth.load_env()
    cano, prdt = _split_account(env.account_no)
    tr_id = _resolve_tr_id(TR_KR_BALANCE, env)
    headers = kis_auth.build_headers(env, tr_id)
    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    url = f"{env.base_url}{BALANCE_PATH}"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        data = r.json()
    except (requests.RequestException, ValueError, AttributeError) as exc:
        log.warning("KIS 잔고조회 실패: %s", exc)
        return Balance(orderable_cash=None, total_eval=None)

    summary = (data.get("output2") or [{}])
    summary0 = summary[0] if summary else {}

    def _f(v: Any) -> float | None:
        try:
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    # inquire-balance 응답엔 ord_psbl_cash 없음 — 가수도정산금액(주문가능 근사),
    # 없으면 예수금총금액으로 폴백. (라이브 검증으로 확인)
    orderable = _f(summary0.get("prvs_rcdl_excc_amt"))
    if orderable is None:
        orderable = _f(summary0.get("dnca_tot_amt"))
    return Balance(
        orderable_cash=orderable,
        total_eval=_f(summary0.get("tot_evlu_amt")),
        holdings=data.get("output1") or [],
        raw=data,
    )
