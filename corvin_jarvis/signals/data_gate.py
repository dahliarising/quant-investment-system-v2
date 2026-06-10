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
        chk = checks.get(str(s.get("symbol", "")))
        if chk and chk.get("flag") == "discrepancy":
            blocked.append({"signal": s, "reason": "price_discrepancy"})
            warnings.append(
                f"⚠️ {s.get('symbol')} 가격 소스 불일치 "
                f"{chk.get('spread_pct', 0):.1f}% — 신호 보류")
            continue
        passed.append(_fix_label(s, market_open))
    return {"passed": passed, "blocked": blocked, "warnings": warnings}
