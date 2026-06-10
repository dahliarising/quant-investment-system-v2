"""만기 신호 채점 — HIT / LATE_HIT / MISS / UNSCORABLE (스펙 §4.2).

score_row는 순수 함수 (가격 주입). run()이 fetch + 채점 + 원장 갱신.
LATE 유예: VELOCITY·방향성 신호는 만기×1.5까지 보류(None 반환) 후 최종 판정.
"""
from __future__ import annotations

from typing import Any

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
        late = closes_after[:int(grace) + 1]
        if any(c <= stop for c in late):
            return "late_hit", {"min_close": min(late), "stop": stop}
        if age < grace:
            return None  # 유예 중
        return "miss", {"min_close": min(closes_after), "stop": stop}

    if kind == "RS_WEAK":
        h_ret = _ret_pct(closes_after[:horizon])
        b_ret = _ret_pct(bench_after[:horizon])
        if h_ret is None or b_ret is None:
            return "unscorable", {"reason": "insufficient closes"}
        rs = h_ret - b_ret
        verdict = "hit" if rs < 0 else "miss"
        return verdict, {"rs_pct": round(rs, 2)}

    if kind == "EVENT":
        if not bench_after or not bench_before or len(bench_before) < 3:
            return "unscorable", {"reason": "no bench data"}
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
        r_grace = _ret_pct(series[:int(grace) + 1])
        late = (r_grace or 0) > 0 if direction == "bull" else (r_grace or 0) < 0
        if late:
            return "late_hit", {"ret_pct": round(r_grace, 2)}
        return "miss", {"ret_pct": round(r, 2)}

    return "unscorable", {"reason": f"no scoring rule for kind={kind}"}
