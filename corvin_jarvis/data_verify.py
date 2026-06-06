"""Phase 0 — 데이터 이중·삼중 교차검증 레이어.

순수 판정(cross_check) + 라이브 다소스 바인딩 분리. 틀린 숫자가 분석에
못 들어오게 — 모든 핵심 수치는 ≥2 독립 소스 교차확인 후 사용.
(폐하 2026-06-06 지시: 증거 데이터 이중·삼중 재검증.)
"""
from __future__ import annotations

import math
import statistics
from typing import Callable


def cross_check(key: str, source_values: dict[str, float | None],
                tol_pct: float) -> dict:
    """여러 소스 값을 교차검증.

    반환: {value, agree, sources, spread_pct, confidence, flag}
    - ≥2 소스 일치(spread ≤ tol) → 중앙값, confidence=high, flag=None
    - ≥2 소스 불일치 → 중앙값(보수), confidence=low, flag="discrepancy"
    - 1 소스 → 그 값, confidence=medium, flag="single_source"
    - 0 소스 → None, confidence=none, flag="missing" (추측 금지)
    """
    present = {s: v for s, v in source_values.items() if v is not None}
    names = list(present.keys())
    vals = list(present.values())

    if not vals:
        return {"value": None, "agree": False, "sources": [],
                "spread_pct": 0.0, "confidence": "none", "flag": "missing"}

    if len(vals) == 1:
        return {"value": vals[0], "agree": None, "sources": names,
                "spread_pct": 0.0, "confidence": "medium", "flag": "single_source"}

    median = statistics.median(vals)
    spread_pct = (max(vals) - min(vals)) / median * 100 if median else 0.0
    agree = spread_pct <= tol_pct
    return {
        "value": median,
        "agree": agree,
        "sources": names,
        "spread_pct": spread_pct,
        "confidence": "high" if agree else "low",
        "flag": None if agree else "discrepancy",
    }


def reconcile_pnl(stored: float | None, live: float | None,
                  tol_pct: float = 3.0) -> dict:
    """저장 pnl vs 라이브 pnl 정합. 괴리 크면 stale 경고, 라이브 우선.

    (TSLA 저장 -3.5% vs 라이브 -10.5% 같은 stale-portfolio 버그 방지.)
    반환: {value, stale, divergence_pp, flag}
    """
    if live is None:
        return {"value": stored, "stale": False, "divergence_pp": 0.0,
                "flag": "no_live"}
    if stored is None:
        return {"value": live, "stale": False, "divergence_pp": 0.0, "flag": None}
    divergence = abs(live - stored)            # percentage-point gap
    stale = divergence > tol_pct
    return {"value": live, "stale": stale, "divergence_pp": divergence,
            "flag": "stored_stale" if stale else None}


def sanity(value, positive: bool = False) -> bool:
    """NaN/inf/None 거부. positive=True면 양수 가격만 허용."""
    if value is None:
        return False
    try:
        if math.isnan(value) or math.isinf(value):
            return False
    except TypeError:
        return False
    if positive and value <= 0:
        return False
    return True


def verified(key: str, fetchers: dict[str, Callable[[], float | None]],
             tol_pct: float = 1.0, positive: bool = False) -> dict:
    """다소스 fetcher를 모두 호출(예외/비정상은 None 처리) → cross_check.

    라이브 IO 경계: fetcher 주입(DI). 한 소스가 죽어도 분석 안 깨짐.
    """
    source_values: dict[str, float | None] = {}
    for name, fetch in fetchers.items():
        try:
            v = fetch()
        except Exception:               # noqa: BLE001 — 소스 격리
            v = None
        source_values[name] = v if sanity(v, positive=positive) else None
    return cross_check(key, source_values, tol_pct)
