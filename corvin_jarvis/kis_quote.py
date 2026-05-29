"""Corvin Jarvis — KIS 시세 fetcher (Tier 2.5)

한국·미국 주식의 현재가 + 일봉 OHLCV를 KIS OpenAPI로 조회.

dca_timing.fetcher 호환 인터페이스 제공:
    fetcher(symbol, days=252) -> list[float]  (oldest → newest closes)

조회 함수:
    get_kr_current_price(symbol) -> float | None
    get_kr_daily_closes(symbol, days=252) -> list[float]
    get_us_current_price(symbol, excd) -> float | None
    get_us_daily_closes(symbol, excd, days=252) -> list[float]

TR_ID:
    국내 현재가:   FHKST01010100
    국내 일봉:     FHKST03010100
    해외 현재가:   HHDFS00000300
    해외 기간시세: HHDFS76240000
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests

from corvin_jarvis import kis_auth

log = logging.getLogger("corvin.kis.quote")

KST = ZoneInfo("Asia/Seoul")

TR_DOMESTIC_PRICE = "FHKST01010100"
TR_DOMESTIC_DAILY = "FHKST03010100"
TR_OVERSEAS_PRICE = "HHDFS00000300"
TR_OVERSEAS_DAILY = "HHDFS76240000"

# 모의투자 일부 TR_ID는 prod와 동일하나 일부는 'V'로 시작. KIS 문서: 시세 조회 TR은 prod ID 그대로.

REQUEST_TIMEOUT = 10
# KIS REST rate limit (실전계좌): 국내 초당 20건, 해외 초당 ~1건.
# tr_id 첫 글자로 분기 — H=Overseas, F=Domestic
_DOMESTIC_TR_PREFIX = "F"
_OVERSEAS_TR_PREFIX = "H"
RATE_LIMIT_DELAY_DOMESTIC = 0.06   # 초당 ~16건 (한도 80%)
RATE_LIMIT_DELAY_OVERSEAS = 0.55   # 초당 ~1.8건 (해외 보수적 — burst safety)
MAX_RETRIES_ON_RATE_LIMIT = 3
RATE_LIMIT_BACKOFF_BASE = 0.8      # 0.8 → 1.6 → 3.2 (exp)


def _delay_for(tr_id: str) -> float:
    return RATE_LIMIT_DELAY_OVERSEAS if tr_id.startswith(_OVERSEAS_TR_PREFIX) else RATE_LIMIT_DELAY_DOMESTIC


_RATE_LIMIT_CODES = frozenset({"EGW00201", "EGW00301"})


def _is_rate_limit_error(data: dict[str, Any]) -> bool:
    """KIS rate limit 코드 식별 — EGW00201(초당), EGW00301(분당)."""
    return str(data.get("msg_cd") or "") in _RATE_LIMIT_CODES


# ============================================================
# Low-level HTTP — retry + exponential backoff on rate limit
# ============================================================


def _http_get(
    env: kis_auth.KISEnv,
    path: str,
    tr_id: str,
    params: dict[str, str],
) -> dict[str, Any]:
    url = f"{env.base_url}{path}"
    headers = kis_auth.build_headers(env, tr_id)
    delay = _delay_for(tr_id)

    last_data: dict[str, Any] = {}
    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
        r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        try:
            data = r.json()
        except (ValueError, AttributeError):
            data = {}
        last_data = data

        # rate limit → exponential backoff + retry
        if r.status_code in (429, 500) and _is_rate_limit_error(data):
            if attempt < MAX_RETRIES_ON_RATE_LIMIT:
                backoff = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                log.info("KIS rate-limit (attempt %d) — sleep %.2fs", attempt + 1, backoff)
                time.sleep(backoff)
                continue
            log.warning("KIS %s rate-limit exhausted after %d retries", tr_id, attempt)
            return {}

        if r.status_code != 200:
            log.warning("KIS %s %s → %d: %s", tr_id, path, r.status_code, r.text[:200])
            return {}

        rt_cd = data.get("rt_cd", "")
        if rt_cd not in ("0", ""):
            log.warning("KIS %s rt_cd=%s msg=%s", tr_id, rt_cd, data.get("msg1", "")[:120])
        time.sleep(delay)
        return data

    return last_data


# ============================================================
# 국내주식 (Korean stocks)
# ============================================================


def get_kr_quote(
    symbol: str,
    env: kis_auth.KISEnv | None = None,
) -> tuple[float | None, float | None]:
    """국내주식 (price, pct_change%). 둘 다 실패면 (None, None)."""
    env = env or kis_auth.load_env()
    data = _http_get(
        env,
        "/uapi/domestic-stock/v1/quotations/inquire-price",
        TR_DOMESTIC_PRICE,
        {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol},
    )
    output = data.get("output") or {}
    raw_price = output.get("stck_prpr")
    raw_pct = output.get("prdy_ctrt")  # 전일대비율
    price: float | None = None
    pct: float | None = None
    try:
        if raw_price not in (None, ""):
            price = float(raw_price)
    except (TypeError, ValueError):
        price = None
    try:
        if raw_pct not in (None, ""):
            pct = float(raw_pct)
    except (TypeError, ValueError):
        pct = None
    return price, pct


def get_kr_current_price(symbol: str, env: kis_auth.KISEnv | None = None) -> float | None:
    """국내주식 현재가만 — backward compatible 단일값."""
    price, _ = get_kr_quote(symbol, env=env)
    return price


def get_kr_fundamentals(
    symbol: str,
    env: kis_auth.KISEnv | None = None,
) -> dict[str, Any]:
    """국내주식 PER/PBR/EPS/BPS/52주고저.

    FHKST01010100(현재가) 응답 output에 이미 포함된 밸류 지표를 파싱.
    pykrx가 누락하는 데이터를 KIS로 보완. output 비면 빈 dict 반환 → 호출자 fallback.
    """
    env = env or kis_auth.load_env()
    data = _http_get(
        env,
        "/uapi/domestic-stock/v1/quotations/inquire-price",
        TR_DOMESTIC_PRICE,
        {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol},
    )
    output = data.get("output") or {}
    if not output:
        return {}

    def _f(key: str) -> float | None:
        raw = output.get(key)
        try:
            if raw not in (None, ""):
                return float(raw)
        except (TypeError, ValueError):
            return None
        return None

    return {
        "symbol": symbol,
        "PER": _f("per"),
        "PBR": _f("pbr"),
        "EPS": _f("eps"),
        "BPS": _f("bps"),
        "52주최고": _f("w52_hgpr"),
        "52주최저": _f("w52_lwpr"),
        "_source": "kis",
    }


def get_kr_daily_closes(
    symbol: str,
    days: int = 252,
    env: kis_auth.KISEnv | None = None,
) -> list[float]:
    """국내주식 일봉 종가 (oldest → newest).

    API는 1회 요청당 최대 100건 반환. days>100이면 fid_input_date_1 sliding로 페이지네이션.
    """
    env = env or kis_auth.load_env()
    end_dt = datetime.now(KST).date()
    target = max(days, 1)
    closes_by_date: dict[str, float] = {}

    cursor_end = end_dt
    # 안전한 상한: 페이지 ~10회 (1000일)
    for _ in range(12):
        cursor_start = cursor_end - timedelta(days=140)  # 100 영업일 약 140 캘린더일 buffer
        data = _http_get(
            env,
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            TR_DOMESTIC_DAILY,
            {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": symbol,
                "fid_input_date_1": cursor_start.strftime("%Y%m%d"),
                "fid_input_date_2": cursor_end.strftime("%Y%m%d"),
                "fid_period_div_code": "D",
                "fid_org_adj_prc": "0",  # 0=수정주가
            },
        )
        rows = data.get("output2") or []
        if not rows:
            break
        for row in rows:
            d = row.get("stck_bsop_date")
            close_raw = row.get("stck_clpr")
            if not d or close_raw in (None, "", "0"):
                continue
            try:
                closes_by_date[d] = float(close_raw)
            except (TypeError, ValueError):
                continue
        if len(closes_by_date) >= target:
            break
        # 더 옛날로 페이지 이동
        cursor_end = cursor_start - timedelta(days=1)

    sorted_dates = sorted(closes_by_date.keys())
    closes = [closes_by_date[d] for d in sorted_dates]
    return closes[-target:]


# ============================================================
# 해외주식 (US stocks)
# ============================================================


def _us_resolve_excd(symbol: str, exchange: str | None) -> str:
    """exchange 명시 없으면 NAS 시도 → 실패 시 caller가 NYS retry."""
    if exchange:
        return exchange.upper()
    return "NAS"


def get_us_quote(
    symbol: str,
    exchange: str | None = None,
    env: kis_auth.KISEnv | None = None,
) -> tuple[float | None, float | None]:
    """해외주식 (price, pct_change%). exchange 미지정 시 NAS→NYS→AMS 순회."""
    env = env or kis_auth.load_env()
    for excd in ([exchange.upper()] if exchange else ["NAS", "NYS", "AMS"]):
        data = _http_get(
            env,
            "/uapi/overseas-price/v1/quotations/price",
            TR_OVERSEAS_PRICE,
            {"AUTH": "", "EXCD": excd, "SYMB": symbol},
        )
        output = data.get("output") or {}
        last = output.get("last")
        rate = output.get("rate")
        if last not in (None, "", "0"):
            try:
                price = float(last)
                pct: float | None = None
                try:
                    if rate not in (None, ""):
                        pct = float(rate)
                except (TypeError, ValueError):
                    pct = None
                return price, pct
            except (TypeError, ValueError):
                continue
    return None, None


def get_us_current_price(
    symbol: str,
    exchange: str | None = None,
    env: kis_auth.KISEnv | None = None,
) -> float | None:
    price, _ = get_us_quote(symbol, exchange=exchange, env=env)
    return price


def get_us_daily_closes(
    symbol: str,
    exchange: str | None = None,
    days: int = 252,
    env: kis_auth.KISEnv | None = None,
) -> list[float]:
    """해외주식 일봉 종가. KIS는 BYMD 기준 30건씩 반환 → 페이지네이션."""
    env = env or kis_auth.load_env()
    target = max(days, 1)
    closes_by_date: dict[str, float] = {}

    excd_candidates = [exchange.upper()] if exchange else ["NAS", "NYS", "AMS"]
    excd_used: str | None = None
    cursor = datetime.now(KST).date()

    # 1차 시도로 거래소 확정
    for excd in excd_candidates:
        params = {
            "AUTH": "",
            "EXCD": excd,
            "SYMB": symbol,
            "GUBN": "0",  # 0=일봉
            "BYMD": cursor.strftime("%Y%m%d"),
            "MODP": "0",
        }
        data = _http_get(
            env,
            "/uapi/overseas-price/v1/quotations/dailyprice",
            TR_OVERSEAS_DAILY,
            params,
        )
        rows = data.get("output2") or []
        if rows:
            excd_used = excd
            for row in rows:
                d = row.get("xymd") or row.get("stck_bsop_date")
                clos = row.get("clos") or row.get("stck_clpr")
                if not d or clos in (None, "", "0"):
                    continue
                try:
                    closes_by_date[d] = float(clos)
                except (TypeError, ValueError):
                    continue
            break

    if excd_used is None:
        return []

    # 추가 페이지: 가장 오래된 날짜 - 1을 BYMD로
    for _ in range(12):
        if len(closes_by_date) >= target:
            break
        if not closes_by_date:
            break
        oldest = min(closes_by_date.keys())
        try:
            cursor_dt = datetime.strptime(oldest, "%Y%m%d").date() - timedelta(days=1)
        except ValueError:
            break
        data = _http_get(
            env,
            "/uapi/overseas-price/v1/quotations/dailyprice",
            TR_OVERSEAS_DAILY,
            {
                "AUTH": "",
                "EXCD": excd_used,
                "SYMB": symbol,
                "GUBN": "0",
                "BYMD": cursor_dt.strftime("%Y%m%d"),
                "MODP": "0",
            },
        )
        rows = data.get("output2") or []
        if not rows:
            break
        before = len(closes_by_date)
        for row in rows:
            d = row.get("xymd") or row.get("stck_bsop_date")
            clos = row.get("clos") or row.get("stck_clpr")
            if not d or clos in (None, "", "0"):
                continue
            try:
                closes_by_date[d] = float(clos)
            except (TypeError, ValueError):
                continue
        if len(closes_by_date) == before:
            break  # 더 이상 새 데이터 없음

    sorted_dates = sorted(closes_by_date.keys())
    closes = [closes_by_date[d] for d in sorted_dates]
    return closes[-target:]


# ============================================================
# Public fetcher (dca_timing 호환)
# ============================================================


def _is_kr_symbol(symbol: str) -> bool:
    return symbol.isdigit() and len(symbol) == 6


def make_fetcher(
    exchange_map: dict[str, str] | None = None,
    env: kis_auth.KISEnv | None = None,
):
    """dca_timing.fetcher 호환 함수 반환.

    exchange_map: {symbol: "NAS"/"NYS"/"AMS"} optional, 없으면 auto-fallback.
    env: 테스트용 — None이면 환경변수에서 lazy load.
    """
    resolved_env = env  # lazy

    def fetcher(symbol: str, days: int = 252) -> list[float]:
        nonlocal resolved_env
        try:
            if resolved_env is None:
                resolved_env = kis_auth.load_env()
        except kis_auth.KISConfigError as e:
            log.warning("KIS unavailable (%s) — caller should fallback", e)
            return []

        if _is_kr_symbol(symbol):
            return get_kr_daily_closes(symbol, days=days, env=resolved_env)
        excd = (exchange_map or {}).get(symbol)
        return get_us_daily_closes(symbol, exchange=excd, days=days, env=resolved_env)

    return fetcher
