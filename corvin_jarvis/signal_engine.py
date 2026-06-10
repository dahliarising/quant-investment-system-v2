"""STAGE① 시그널 엔진 — 명시적 가격 손절선 + 근접 시그널 + 통합 랭킹.

기존 로직을 재구축하지 않는다. 빠진 한 조각 = **명시적 가격 손절선**(종가기준)을
codify하고, 보유 포지션을 긴급도순으로 랭킹한 dashboard/페이퍼봇용 시그널 리스트를 낸다.

- PnL 기반 손절/익절 임계는 `signals.verdict`에서 재사용 (단일 출처).
- DCA/진입존 시그널은 `playbook.builder`가 이미 담당 → 여기선 보유분 관리에 집중.
- 순수 함수 · 부작용 없음 · 실주문 0 (STAGE②까지 안전구간).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corvin_jarvis.signals.verdict import _STOP_LOSS, _TAKE_PROFIT

# 손절선 위 이 비율 이내면 '근접(WATCH)'. 종가기준 손절 운용과 정합 (장중 이탈 ≠ 발동).
NEAR_STOP_PCT = 3.0

_STOPS_PATH = Path(__file__).resolve().parent / "stops.json"


def _fmt(v: float) -> str:
    """가격 가독 포맷 — 큰 수(KR 원)는 천단위, 작은 수(USD)는 소수 유지. 1e+06 방지."""
    if v >= 10000:
        return f"{round(v):,}"
    return f"{v:g}"


@dataclass(frozen=True)
class EngineSignal:
    symbol: str
    kind: str                       # STOP | WATCH | TRIM | HOLD | UNKNOWN
    action: str                     # 손절검토 | 관찰 | 비중축소 | 홀딩 | —
    urgency: int                    # 0-100, 랭킹 키
    reason: str
    price: float | None
    pnl_pct: float | None
    stop: float | None
    stop_distance_pct: float | None  # (price/stop-1)*100, 음수=손절선 아래

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "kind": self.kind, "action": self.action,
            "urgency": self.urgency, "reason": self.reason, "price": self.price,
            "pnl_pct": self.pnl_pct, "stop": self.stop,
            "stop_distance_pct": self.stop_distance_pct,
        }


def load_stops(path: Path | None = None) -> dict[str, float]:
    """명시적 가격 손절선 로드 ({symbol: stop_price}). 파일 없으면 빈 dict."""
    p = path or _STOPS_PATH
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, float] = {}
    for sym, val in raw.items():
        if sym.startswith("_"):  # _note 등 메타키 스킵
            continue
        try:
            out[str(sym)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _eval_one(pos: dict[str, Any], stops: dict[str, float],
              take_profit_pct: float, near_stop_pct: float) -> EngineSignal:
    sym = str(pos.get("symbol", ""))
    price = pos.get("price")
    pnl = pos.get("pnl_pct")
    stop = stops.get(sym)
    dist = (price / stop - 1) * 100 if (price and stop) else None

    def mk(kind: str, action: str, urgency: int, reason: str) -> EngineSignal:
        return EngineSignal(symbol=sym, kind=kind, action=action, urgency=urgency,
                            reason=reason, price=price, pnl_pct=pnl, stop=stop,
                            stop_distance_pct=round(dist, 2) if dist is not None else None)

    if price is None:
        return mk("UNKNOWN", "—", 0, "가격 미확보 — 평가 불가 (앱 확인)")

    # 1) 명시 손절선 종가 이탈 = 최우선
    if stop is not None and price <= stop:
        return mk("STOP", "손절검토", 95,
                  f"종가 {_fmt(price)} ≤ 손절선 {_fmt(stop)} 이탈 — 종가기준 발동 검토")
    # 2) 손절선 근접 (위)
    if dist is not None and 0 <= dist <= near_stop_pct:
        return mk("WATCH", "관찰", 70,
                  f"손절선 {_fmt(stop)} +{dist:.1f}% 근접 — 종가 주시")
    # 3) 명시선 없을 때 PnL 손절
    if stop is None and pnl is not None and pnl <= _STOP_LOSS:
        return mk("STOP", "손절검토", 80,
                  f"PnL {pnl:.1f}% ≤ {_STOP_LOSS:g}% — 명시 손절선 미설정, 검토")
    # 4) 익절선 도달
    if pnl is not None and pnl >= take_profit_pct:
        return mk("TRIM", "비중축소", 55,
                  f"PnL +{pnl:.1f}% ≥ +{take_profit_pct:g}% — 일부 차익실현 검토")
    # 5) 정상 보유
    tail = f" · 손절선 {_fmt(stop)}(+{dist:.1f}%)" if dist is not None else ""
    return mk("HOLD", "홀딩", 20, f"보유 논리 유효{tail}")


def evaluate(holdings: list[dict[str, Any]], *,
             stops: dict[str, float] | None = None,
             take_profit_pct: float = _TAKE_PROFIT,
             near_stop_pct: float = NEAR_STOP_PCT) -> list[EngineSignal]:
    """보유 포지션 리스트 → 긴급도 내림차순 시그널 리스트 (advisory·모의)."""
    stops = stops if stops is not None else load_stops()
    sigs = [_eval_one(p, stops, take_profit_pct, near_stop_pct) for p in holdings]
    return sorted(sigs, key=lambda s: s.urgency, reverse=True)
