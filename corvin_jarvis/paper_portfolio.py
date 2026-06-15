"""현금관리 가상 포트폴리오 — 페이퍼트레이딩 '공격쪽' 엔진 (매수 진입 포함).

기존 paper_trader.py(매도 신호→가상청산→실현손익)와 별개. 이건 현금+포지션을
든 *살아있는 가상 계좌*로, 예측기반 매수 진입까지 회계한다. 전략(선정/가드/사이징)은
상위 레이어; 여기선 체결 회계 + mark-to-market + 비중만 (순수·불변).

🚨 실주문 0. 통화: KR+US 통합 성과를 위해 KRW 기준 단일 계좌(미국 체결은 호출자가
FX 환산해 price_krw로 전달). 통화는 holding.currency로 태깅(리포팅용).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

DEFAULT_INITIAL_KRW = 10_000_000


@dataclass(frozen=True)
class Holding:
    qty: int
    avg_price_krw: float
    currency: str = "KRW"


@dataclass(frozen=True)
class PaperPortfolio:
    cash_krw: float
    initial_krw: float
    holdings: Mapping[str, Holding] = field(default_factory=dict)
    realized_pnl_krw: float = 0.0
    trades: tuple[dict[str, Any], ...] = ()


def _record(pf: PaperPortfolio, symbol: str, side: str, qty: int,
            price_krw: float, ts: str, reason: str) -> dict[str, Any]:
    return {"ts": ts, "symbol": symbol, "side": side, "qty": qty,
            "price_krw": price_krw, "reason": reason}


def buy(pf: PaperPortfolio, symbol: str, qty: int, price_krw: float, *,
        currency: str = "KRW", ts: str = "", reason: str = "") -> PaperPortfolio:
    """매수 진입 — 가중평균 갱신. 현금 부족 시 ValueError (불변 반환)."""
    if qty <= 0:
        raise ValueError(f"수량은 양수여야 함: {qty}")
    cost = qty * price_krw
    if cost > pf.cash_krw + 1e-6:
        raise ValueError(f"현금 부족: 필요 {cost:,.0f} > 보유 {pf.cash_krw:,.0f}")
    old = pf.holdings.get(symbol)
    if old:
        new_qty = old.qty + qty
        new_avg = (old.qty * old.avg_price_krw + qty * price_krw) / new_qty
        h = Holding(new_qty, new_avg, old.currency)
    else:
        h = Holding(qty, price_krw, currency)
    return replace(
        pf,
        cash_krw=pf.cash_krw - cost,
        holdings={**pf.holdings, symbol: h},
        trades=(*pf.trades, _record(pf, symbol, "buy", qty, price_krw, ts, reason)),
    )


def sell(pf: PaperPortfolio, symbol: str, qty: int, price_krw: float, *,
         ts: str = "", reason: str = "") -> PaperPortfolio:
    """매도 청산 — 실현손익 누적. 보유 초과/미보유 시 ValueError (불변 반환)."""
    old = pf.holdings.get(symbol)
    if not old or qty > old.qty:
        raise ValueError(f"매도 불가: {symbol} 보유 {old.qty if old else 0} < 요청 {qty}")
    realized = qty * (price_krw - old.avg_price_krw)
    new_holdings = dict(pf.holdings)
    if qty == old.qty:
        del new_holdings[symbol]
    else:
        new_holdings[symbol] = Holding(old.qty - qty, old.avg_price_krw, old.currency)
    return replace(
        pf,
        cash_krw=pf.cash_krw + qty * price_krw,
        holdings=new_holdings,
        realized_pnl_krw=pf.realized_pnl_krw + realized,
        trades=(*pf.trades, _record(pf, symbol, "sell", qty, price_krw, ts, reason)),
    )


def mark_to_market(pf: PaperPortfolio, prices_krw: Mapping[str, float]) -> dict[str, Any]:
    """현재가(KRW)로 평가. 가격 없는 종목은 취득원가로 보수 평가."""
    holdings_value = 0.0
    detail = []
    for sym, h in pf.holdings.items():
        px = prices_krw.get(sym, h.avg_price_krw)
        val = h.qty * px
        holdings_value += val
        upnl = h.qty * (px - h.avg_price_krw)
        detail.append({
            "symbol": sym, "qty": h.qty, "avg_krw": h.avg_price_krw,
            "price_krw": px, "value_krw": val, "upnl_krw": upnl,
            "upnl_pct": (px / h.avg_price_krw - 1) * 100 if h.avg_price_krw else 0.0,
            "currency": h.currency,
        })
    total = pf.cash_krw + holdings_value
    pnl = total - pf.initial_krw
    return {
        "total_value_krw": total,
        "cash_krw": pf.cash_krw,
        "holdings_value_krw": holdings_value,
        "realized_pnl_krw": pf.realized_pnl_krw,
        "pnl_krw": pnl,
        "pnl_pct": (pnl / pf.initial_krw * 100) if pf.initial_krw else 0.0,
        "positions": detail,
    }


def position_weight(pf: PaperPortfolio, symbol: str,
                    prices_krw: Mapping[str, float]) -> float:
    """종목 비중 % (총자산 대비) — 리스크 상한 가드용."""
    snap = mark_to_market(pf, prices_krw)
    total = snap["total_value_krw"]
    if not total:
        return 0.0
    h = pf.holdings.get(symbol)
    if not h:
        return 0.0
    px = prices_krw.get(symbol, h.avg_price_krw)
    return h.qty * px / total * 100


# ============================================================
# 영속화
# ============================================================

def save(pf: PaperPortfolio, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cash_krw": pf.cash_krw,
        "initial_krw": pf.initial_krw,
        "realized_pnl_krw": pf.realized_pnl_krw,
        "holdings": {s: {"qty": h.qty, "avg_price_krw": h.avg_price_krw,
                         "currency": h.currency} for s, h in pf.holdings.items()},
        "trades": list(pf.trades),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load(path: Path, initial_krw: float = DEFAULT_INITIAL_KRW) -> PaperPortfolio:
    """저장된 포트폴리오 로드. 없으면 초기자본 fresh."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return PaperPortfolio(cash_krw=initial_krw, initial_krw=initial_krw)
    holdings = {
        s: Holding(h["qty"], h["avg_price_krw"], h.get("currency", "KRW"))
        for s, h in (data.get("holdings") or {}).items()
    }
    return PaperPortfolio(
        cash_krw=data.get("cash_krw", initial_krw),
        initial_krw=data.get("initial_krw", initial_krw),
        holdings=holdings,
        realized_pnl_krw=data.get("realized_pnl_krw", 0.0),
        trades=tuple(data.get("trades") or ()),
    )
