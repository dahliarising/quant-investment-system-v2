"""Phase 4 — 데이터 검증 게이트. 신호 발화·기록 전 자동 검증 (스펙 §7).

ledger.record 직전 훅: 통과 신호만 기록·발화. 순수 판정(gate_signals) +
라이브 다소스 바인딩(collect_price_checks) 분리 — data_verify.py 패턴.
"""
from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("corvin.signals.data_gate")

_PRICE_TOL_PCT = 1.0     # KIS vs yahoo/pykrx 허용 괴리 (스펙 §7: ±1%)
_MAX_STALE_DAYS = 3      # 데이터 경과 기본 한도 — 초과 시 신호 차단
_NUMERIC_FIELDS = ("urgency", "confidence", "horizon_days")


def _numbers_ok(sig: dict[str, Any]) -> bool:
    """핵심 수치 필드 NaN/inf/비수치 거부. None은 허용(옵셔널 필드)."""
    for f in _NUMERIC_FIELDS:
        v = sig.get(f)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if math.isnan(fv) or math.isinf(fv):
            return False
    return True


def _fix_label(sig: dict[str, Any], market_open: bool) -> dict[str, Any]:
    """장마감이면 '현재가' → '전일종가' 교정. 새 dict 반환 (비변이)."""
    msg = str(sig.get("message", ""))
    if market_open or "현재가" not in msg:
        return sig
    return {**sig, "message": msg.replace("현재가", "전일종가")}


def gate_signals(signals: list[dict[str, Any]], *,
                 price_checks: dict[str, dict] | None = None,
                 market_open: bool = True,
                 data_age_days: int | None = None,
                 max_age_days: int = _MAX_STALE_DAYS) -> dict[str, Any]:
    """신호 리스트 검증 — {passed, blocked, warnings} 반환.

    - data_age_days > max_age_days → 전체 차단 (stale_data) + 갱신 요청
    - NaN/inf 수치 → 해당 신호 drop (invalid_numeric)
    - price_checks[symbol].flag == "discrepancy" → 보류 (price_discrepancy) + 경고
    - market_open=False → message의 '현재가' → '전일종가' 교정
    """
    checks = price_checks or {}
    if data_age_days is not None and data_age_days > max_age_days:
        return {
            "passed": [],
            "blocked": [{"signal": s, "reason": "stale_data"} for s in signals],
            "warnings": [f"⚠️ 데이터 {data_age_days}일 경과 — "
                         f"신호 {len(signals)}건 차단, 갱신 필요"],
        }
    passed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    warnings: list[str] = []
    for s in signals:
        if not _numbers_ok(s):
            blocked.append({"signal": s, "reason": "invalid_numeric"})
            continue
        sym_key = str(s.get("symbol") or "")
        chk = checks.get(sym_key) if sym_key else None
        if chk and chk.get("flag") == "discrepancy":
            blocked.append({"signal": s, "reason": "price_discrepancy"})
            warnings.append(
                f"⚠️ {s.get('symbol')} 가격 소스 불일치 "
                f"{chk.get('spread_pct', 0):.1f}% — 신호 보류")
            continue
        # 얕은 복사 — passed 별칭 변이가 원본 신호 리스트 오염 방지
        passed.append(_fix_label(dict(s), market_open))
    return {"passed": passed, "blocked": blocked, "warnings": warnings}


# ── 라이브 다소스 바인딩 ──────────────────────────────────

def _kis_price(symbol: str) -> float | None:
    """KIS 시세 (KR/US 자동 분기). env 없으면 None."""
    from corvin_jarvis import kis_quote
    from corvin_jarvis import quote_provider as qp
    env = qp._get_kis_env()
    if env is None:
        return None
    if qp.is_kr_stock(symbol):
        price, _ = kis_quote.get_kr_quote(symbol, env=env)
    else:
        price, _ = kis_quote.get_us_quote(
            symbol, exchange=qp._exchange_for(symbol), env=env)
    return price


def _alt_price(symbol: str) -> float | None:
    """교차 소스 — KR: pykrx (yfinance 금지 규칙), US: yfinance."""
    from corvin_jarvis import quote_provider as qp
    if qp.is_kr_stock(symbol):
        from kr_data import get_kr_stock_data  # noqa: PLC0415
        data = get_kr_stock_data(symbol)
        return None if "에러" in data else float(data["현재가"])
    return qp._yfinance_quote(symbol).price


_CHECK_CACHE: dict[str, tuple[float, dict]] = {}   # sym -> (expires_at, result)
_CHECK_TTL = 300.0   # 라이브 교차검증 캐시 — snapshot 30s TTL 미스마다 풀 fan-out 방지


def collect_price_checks(symbols: list[str],
                         fetchers_for=None,
                         tol_pct: float = _PRICE_TOL_PCT) -> dict[str, dict]:
    """종목별 KIS vs 교차소스 검증 결과 — gate_signals price_checks 입력.

    fetchers_for(sym) -> {name: fetch} 주입 가능 (테스트/커스텀, 캐시 우회).
    기본(라이브) 경로는 _CHECK_TTL 캐시 — 심볼당 네트워크 2콜 fan-out 억제.
    data_verify.verified가 예외·NaN을 소스 격리 처리.
    """
    import time

    from corvin_jarvis import data_verify
    out: dict[str, dict] = {}
    now = time.time()
    for sym in dict.fromkeys(symbols):       # 순서 보존 dedup
        if fetchers_for is not None:
            out[sym] = data_verify.verified(sym, fetchers_for(sym),
                                            tol_pct=tol_pct, positive=True)
            continue
        cached = _CHECK_CACHE.get(sym)
        if cached and cached[0] > now:
            out[sym] = cached[1]
            continue
        fetchers = {"kis": lambda s=sym: _kis_price(s),
                    "alt": lambda s=sym: _alt_price(s)}
        res = data_verify.verified(sym, fetchers, tol_pct=tol_pct, positive=True)
        _CHECK_CACHE[sym] = (now + _CHECK_TTL, res)
        out[sym] = res
    return out
