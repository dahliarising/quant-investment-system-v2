"""만기 신호 채점 — hit / late_hit / miss / unscorable (스펙 §4.2).

score_row는 순수 함수 (가격 주입). run()이 fetch + 채점 + 원장 갱신.
LATE 유예: VELOCITY·방향성 신호는 만기×1.5까지 보류(None 반환) 후 최종 판정.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("corvin.signals.scorer")

_EVENT_VOL_MULT = 1.3   # 이벤트일 변동 > 직전 평균 ×1.3 → HIT
_GRACE_MULT = 1.5       # late_hit 유예 배수

Verdict = tuple[str, dict[str, Any]]


def _ret_pct(closes: list[float]) -> float | None:
    if len(closes) < 2 or closes[0] <= 0:
        return None
    return (closes[-1] / closes[0] - 1) * 100.0


def score_row(row: dict[str, Any], *,
              closes_after: list[float],
              bench_after: list[float],
              bench_before: list[float] | None = None) -> Verdict | None:
    """단일 만기 신호 채점. None = late 유예 중 (다음 채점까지 open 유지).

    closes_after/bench_after: 발화일 이후 종가 oldest→newest.
    bench_before: EVENT 채점용 — 발화 직전 벤치마크 종가.

    closes_after/bench_after는 거래일 시계열 — 호출자(러너)가 달력일 age를
    거래일 수로 변환해 슬라이스해서 전달할 책임 (horizon은 원소 개수로 사용됨).
    """
    kind = str(row["kind"])
    horizon = int(row.get("horizon_days") or 10)
    age = int(row.get("age_days") or horizon)
    ev = row.get("evidence") or {}
    grace = horizon * _GRACE_MULT

    if kind == "VELOCITY":
        stop = ev.get("stop")
        if stop is None or not closes_after:
            return "unscorable", {"reason": "no stop or closes"}
        within = closes_after[:horizon]
        if any(c <= stop for c in within):
            return "hit", {"min_close": min(within), "stop": stop}
        late = closes_after[:int(grace)]
        if any(c <= stop for c in late):
            return "late_hit", {"min_close": min(late), "stop": stop}
        if age < grace:
            return None  # 유예 중
        return "miss", {"min_close": min(closes_after), "stop": stop}

    if kind in ("RS_WEAK", "RS_REVERT"):
        h_ret = _ret_pct(closes_after[:horizon])
        b_ret = _ret_pct(bench_after[:horizon])
        if h_ret is None or b_ret is None:
            return "unscorable", {"reason": "insufficient closes"}
        rs = h_ret - b_ret
        # RS_WEAK=약세 지속(rs<0) hit / RS_REVERT=반등(rs>0) hit — 역발상
        hit = (rs < 0) if kind == "RS_WEAK" else (rs > 0)
        return ("hit" if hit else "miss"), {"rs_pct": round(rs, 2)}

    if kind == "EVENT":
        if not bench_after or not bench_before or len(bench_before) < 3:
            return "unscorable", {"reason": "no bench data"}
        if any(p <= 0 for p in bench_before):
            return "unscorable", {"reason": "zero or negative bench price"}
        prior_moves = [abs(bench_before[i] / bench_before[i - 1] - 1)
                       for i in range(1, len(bench_before))]
        avg_move = sum(prior_moves) / len(prior_moves)
        event_move = abs(bench_after[0] / bench_before[-1] - 1)
        verdict = "hit" if (avg_move > 0 and event_move > avg_move * _EVENT_VOL_MULT) else "miss"
        return verdict, {"event_move_pct": round(event_move * 100, 2),
                         "avg_move_pct": round(avg_move * 100, 2)}

    if kind in ("STOP", "WATCH"):
        ref = ev.get("price")
        if ref is None or not closes_after:
            return "unscorable", {"reason": "no ref price or closes"}
        low = min(closes_after[:horizon])
        verdict = "hit" if low < ref else "miss"
        return verdict, {"ref_price": ref, "min_close": low}

    direction = row.get("direction")
    if direction in ("bull", "bear"):
        series = closes_after if row.get("symbol") else bench_after
        r = _ret_pct(series[:horizon])
        if r is None:
            return "unscorable", {"reason": "insufficient closes"}
        moved = r > 0 if direction == "bull" else r < 0
        if moved:
            return "hit", {"ret_pct": round(r, 2)}
        if age < grace:
            return None  # 방향성 신호도 유예
        r_grace = _ret_pct(series[:int(grace)])
        late = (r_grace or 0) > 0 if direction == "bull" else (r_grace or 0) < 0
        if late:
            return "late_hit", {"ret_pct": round(r_grace, 2)}
        return "miss", {"ret_pct": round(r, 2)}

    return "unscorable", {"reason": f"no scoring rule for kind={kind}"}


def _trading_days(age_days: int) -> int:
    """달력일 → 거래일 근사 (주 5일). ±1~2일 오차 허용 — 스펙 §4.2 합의."""
    return max(1, round(age_days * 5 / 7))


def _default_fetch(symbol: str, days: int) -> list[float]:
    from corvin_jarvis import quote_provider as qp
    return qp.get_stock_daily_closes(symbol, days=days, completed_only=True)


def run(db_path=None, now=None,
        fetch_closes: Callable[[str, int], list[float]] | None = None) -> dict[str, Any]:
    """만기 신호 일괄 채점. 반환: {"scored": n, "pending": n, "by_status": {...}}."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from corvin_jarvis.signals import ledger

    kst = ZoneInfo("Asia/Seoul")
    t = now or datetime.now(kst)
    fetch = fetch_closes or _default_fetch
    due = ledger.fetch_due(db_path=db_path, now=t)

    by_status: dict[str, int] = {}
    scored = pending = 0
    bench_cache: dict[str, list[float]] = {}

    def bench_for(symbol: str) -> str:
        # 069500 = KODEX200 ETF (KOSPI 프록시 — pykrx/KIS 모두 조회 가능)
        return "069500" if symbol.endswith(".KS") or (symbol.isdigit() and len(symbol) == 6) else "SPY"

    for row in due:
        sym = row["symbol"]
        age = row["age_days"]
        n_days = _trading_days(age)
        try:
            closes_after: list[float] = []
            if sym:
                full = fetch(sym, n_days + 30)
                if not full:
                    # 일시 결손/상폐 가능 — 이번 사이클 skip, open 유지 (영구 unscorable 방지)
                    pending += 1
                    continue
                closes_after = full[-n_days:]
            bkey = bench_for(sym or "SPY")
            if bkey not in bench_cache:
                bench_cache[bkey] = fetch(bkey, n_days + 30) or []
        except Exception as e:  # noqa: BLE001 — fetch 실패는 이 행만 skip
            log.warning("scorer: fetch failed for %s: %s", sym or "(macro)", e)
            by_status["fetch_error"] = by_status.get("fetch_error", 0) + 1
            continue
        bench_full = bench_cache[bkey]
        bench_after = bench_full[-n_days:] if bench_full else []
        bench_before = bench_full[:-n_days][-21:] if len(bench_full) > n_days else []

        needs_bench = row["kind"] in ("RS_WEAK", "EVENT") or (
            row.get("direction") in ("bull", "bear") and not sym)
        if needs_bench and (not bench_after or (row["kind"] == "EVENT" and not bench_before)):
            pending += 1  # 벤치 데이터 일시 결손 — open 유지
            continue

        verdict = score_row(row, closes_after=closes_after,
                            bench_after=bench_after, bench_before=bench_before)
        if verdict is None:
            pending += 1
            continue
        status, outcome = verdict
        ledger.mark_scored(row["id"], status, outcome, db_path=db_path, now=t)
        by_status[status] = by_status.get(status, 0) + 1
        scored += 1

    return {"scored": scored, "pending": pending, "by_status": by_status}


if __name__ == "__main__":
    import json as _json
    res = run()
    from corvin_jarvis.signals import calibration
    calibration.write_state()
    print(_json.dumps(res, ensure_ascii=False))
