"""자동매매 전략 레이어 — 종목선정 + 가드 (행동·리스크).

예측/verdict가 낸 매수·매도 신호를, 오늘 진단한 투자심리 실패를 막는 가드로
거른 뒤 실행 계획(plan)을 낸다. 순수 함수 — 데이터는 주입.

가드 (이번 phase = 핵심 2개):
  ① 행동 no-chase  — 이미 과확장/과매수/고점근접 종목 추격 매수 차단 (FOMO 방지)
  ② 리스크 비중상한 — 종목당 max_position_pct 넘는 매수 수량 캡 (집중 방지)
다음 phase: 캘리브레이션 게이트(검증된 신호만), 섹터 상한.

매도는 보유분이 있어야만 계획 (없는 걸 못 팜). 사이징: 매수=고정 트랜치(buy_krw),
비중상한으로 캡. 분할매수=트랜치 절반.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from corvin_jarvis import paper_portfolio as pp

# 행동 가드 임계 (조정 가능)
MAX_PCT_ABOVE_MA20 = 15.0   # MA20 대비 +15% 이상 = 과확장
MAX_RSI = 70.0              # RSI 70 이상 = 과매수
NEAR_HIGH_PCT = 3.0        # 52주고 -3% 이내 = 거의 고점

SPLIT_BUY_RATIO = 0.5
BUY_ACTIONS = frozenset({"매수", "비중확대"})
SPLIT_BUY_ACTIONS = frozenset({"분할매수"})


@dataclass(frozen=True)
class ChaseMetrics:
    pct_above_ma20: float | None
    rsi: float | None
    pct_from_52w_high: float | None  # (price/high-1)*100, 0=고점, 음수=아래


def is_chasing(m: ChaseMetrics, *,
               max_above_ma: float = MAX_PCT_ABOVE_MA20,
               max_rsi: float = MAX_RSI,
               near_high_pct: float = NEAR_HIGH_PCT) -> tuple[bool, str]:
    """추격 매수인가? (FOMO 가드). 지표 None은 차단 안 함(보수)."""
    if m.pct_above_ma20 is not None and m.pct_above_ma20 > max_above_ma:
        return True, f"MA20 대비 +{m.pct_above_ma20:.0f}% 과확장"
    if m.rsi is not None and m.rsi > max_rsi:
        return True, f"RSI {m.rsi:.0f} 과매수"
    if m.pct_from_52w_high is not None and m.pct_from_52w_high >= -near_high_pct:
        return True, f"52주고 {m.pct_from_52w_high:+.0f}% 거의 고점"
    return False, ""


def cap_qty_to_weight(pf: pp.PaperPortfolio, symbol: str, qty: int,
                      price_krw: float, prices_krw: Mapping[str, float],
                      *, max_pct: float) -> int:
    """비중 상한 넘지 않게 매수 수량 캡. 현금 한도도 반영. 0이면 매수 불가."""
    marks = {**prices_krw, symbol: price_krw}
    snap = pp.mark_to_market(pf, marks)
    total = snap["total_value_krw"] or pf.cash_krw
    held = pf.holdings.get(symbol)
    current_value = (held.qty * price_krw) if held else 0.0
    room_value = max_pct / 100 * total - current_value
    if room_value <= 0:
        return 0
    affordable = int(min(room_value, pf.cash_krw) // price_krw)
    return max(0, min(qty, affordable))


def _planned(symbol: str, side: str, qty: int, price_krw: float, reason: str,
             status: str, skip_reason: str = "") -> dict[str, Any]:
    return {"symbol": symbol, "side": side, "qty": qty, "price_krw": price_krw,
            "reason": reason, "status": status, "skip_reason": skip_reason}


def plan_cycle(
    pf: pp.PaperPortfolio,
    *,
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    prices_krw: Mapping[str, float],
    chase_metrics: Mapping[str, ChaseMetrics],
    buy_krw: float,
    max_position_pct: float,
) -> list[dict[str, Any]]:
    """한 사이클의 실행 계획. 매도(보유분) 먼저, 가드된 매수 다음.

    buy_signals = [{symbol, action}], sell_signals = [{symbol, qty, reason}].
    반환 = planned/skipped dict 리스트 (status·skip_reason 포함, 가시성).
    """
    plans: list[dict[str, Any]] = []

    # 1) 매도 — 보유분만
    for s in sell_signals:
        sym = s["symbol"]
        held = pf.holdings.get(sym)
        if not held:
            continue
        qty = min(int(s.get("qty", held.qty)), held.qty)
        if qty < 1:
            continue
        px = prices_krw.get(sym, held.avg_price_krw)
        plans.append(_planned(sym, "sell", qty, px, s.get("reason", "매도신호"), "planned"))

    # 2) 매수 — 가드 통과분만
    for b in buy_signals:
        sym = b["symbol"]
        action = b.get("action", "매수")
        px = prices_krw.get(sym)
        if not px:
            plans.append(_planned(sym, "buy", 0, 0, action, "skipped", "현재가 없음"))
            continue
        # 행동 가드: 추격 차단
        chasing, why = is_chasing(chase_metrics.get(sym, ChaseMetrics(None, None, None)))
        if chasing:
            plans.append(_planned(sym, "buy", 0, px, action, "skipped", f"추격차단: {why}"))
            continue
        # 사이징 + 리스크 상한
        tranche = buy_krw * (SPLIT_BUY_RATIO if action in SPLIT_BUY_ACTIONS else 1.0)
        want = int(tranche // px)
        qty = cap_qty_to_weight(pf, sym, want, px, prices_krw, max_pct=max_position_pct)
        if qty < 1:
            reason = "비중상한/현금부족" if want >= 1 else "트랜치<1주"
            plans.append(_planned(sym, "buy", 0, px, action, "skipped", reason))
            continue
        plans.append(_planned(sym, "buy", qty, px, f"{action}(가드통과)", "planned"))

    return plans
