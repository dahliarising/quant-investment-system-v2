"""STAGE② 페이퍼 트레이딩 — 모의 청산 + 가상 원장 + 실현손익.

STAGE①(signal_engine)의 손절선/익절선 시그널을 입력으로, "그 가격에 팔았다면"의
모의 체결을 가상 원장(paper_trades.json)에 기록하고 실현손익을 집계한다.

🚨 실주문 0. 하드라인(③ 실주문 자동화) 이전 안전구간. 통화 혼합 방지 위해
실현손익은 종목별 네이티브 통화로만 집계한다 (USD/KRW를 합산하지 않음).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_LEDGER_PATH = Path(__file__).resolve().parent / "paper_trades.json"

# 시그널 kind → 모의 청산 대상. ADD(분할매수)는 STAGE②.1에서.
_EXIT_KINDS = {"STOP", "TRIM"}


@dataclass(frozen=True)
class PaperTrade:
    ts: str
    symbol: str
    side: str       # BUY | SELL
    qty: float
    price: float
    reason: str
    kind: str       # STOP | TRIM | ADD | MANUAL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def simulate_from_signal(sig: Any, position: dict[str, Any], *,
                         ts: str = "", fraction: float = 0.5) -> PaperTrade | None:
    """엔진 시그널 → 모의 청산 1건 (없으면 None).

    STOP = 전량 청산, TRIM = `fraction` 비율 청산(정수주 내림). 소수주 0이면 미체결.
    HOLD/WATCH/UNKNOWN 또는 가격 미확보 시 None (행동 없음).
    """
    if sig.kind not in _EXIT_KINDS or sig.price is None:
        return None
    shares = position.get("shares") or 0
    qty = shares if sig.kind == "STOP" else int(shares * fraction)
    if qty <= 0:
        return None
    return PaperTrade(ts=ts, symbol=sig.symbol, side="SELL", qty=qty,
                      price=float(sig.price), reason=sig.reason, kind=sig.kind)


def realized_pnl(trades: list[PaperTrade],
                 avg_prices: dict[str, float]) -> dict[str, dict[str, Any]]:
    """SELL 체결의 종목별 실현손익(네이티브 통화). avg_prices = 종목별 평단."""
    out: dict[str, dict[str, Any]] = {}
    for t in trades:
        if t.side != "SELL":
            continue
        avg = avg_prices.get(t.symbol)
        if avg is None:
            continue
        d = out.setdefault(t.symbol, {"realized": 0.0, "qty": 0, "trades": 0})
        d["realized"] += t.qty * (t.price - avg)
        d["qty"] += t.qty
        d["trades"] += 1
    for sym, d in out.items():
        cost = d["qty"] * avg_prices[sym]
        d["realized_pct"] = (d["realized"] / cost * 100) if cost else None
    return out


def propose_exits(holdings: list[dict[str, Any]], *, ts: str = "",
                  fraction: float = 0.5,
                  stops: dict[str, float] | None = None) -> list[PaperTrade]:
    """라이브 보유분 → 룰 트리거(STOP/TRIM) 모의 청산 '제안' (기록 안 함).

    holdings 항목 = {symbol, market, price, pnl_pct, shares}. signal_engine로
    시그널을 내고 actionable 한 것만 모의 체결로 변환. 제안일 뿐 원장 미반영
    (실제 기록은 폐하 승인 후 PaperLedger.add).
    """
    from corvin_jarvis import signal_engine as se
    by_sym = {h.get("symbol"): h for h in holdings}
    proposals: list[PaperTrade] = []
    for sig in se.evaluate(holdings, stops=stops):
        pos = by_sym.get(sig.symbol, {})
        t = simulate_from_signal(sig, {"shares": pos.get("shares", 0)},
                                 ts=ts, fraction=fraction)
        if t is not None:
            proposals.append(t)
    return proposals


@dataclass
class PaperLedger:
    """가상 체결 원장 — paper_trades.json 영속화. add는 불변 append."""
    path: Path
    trades: list[PaperTrade] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None = None) -> PaperLedger:
        p = Path(path) if path is not None else _LEDGER_PATH
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = []
        trades = [PaperTrade(**r) for r in raw if isinstance(r, dict)]
        return cls(path=p, trades=trades)

    def add(self, trade: PaperTrade) -> None:
        # 불변 append — 기존 리스트를 변형하지 않고 새 리스트로 교체
        self.trades = [*self.trades, trade]

    def save(self) -> None:
        self.path.write_text(
            json.dumps([t.to_dict() for t in self.trades], ensure_ascii=False, indent=2),
            encoding="utf-8")
