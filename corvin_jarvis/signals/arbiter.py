"""Phase 2 — 통합 중재자. 5엔진 상충 신호 → 종목별 단일 최종 액션 (스펙 §5).

중재 규칙 (우선순위 순):
  ① 안전 우선 — defensive(STOP/hardstop류)는 어떤 매수 신호보다 우선.
  ② 적중률 가중 — buy vs warn 상충 시 calibration hit_rate(n>=10) 우세 쪽.
     양쪽 모두 미검증이면 보수적 '보류'.
  ③ 그 외 — trim > buy > warn > hold.

순수 함수 · 부작용 없음. 입력은 arbiter_inputs가 normalize한 공통 dict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

_MIN_SAMPLES = 10  # MUST match calibration._MIN_SAMPLES — 변경 시 양쪽 동시 수정
_VELOCITY_DEFENSIVE_URGENCY = 70

_ACTION_PRIORITY = {"매도검토": 95, "비중축소": 70, "보류": 60,
                    "매수후보": 50, "관찰": 40, "홀딩": 20}


@dataclass(frozen=True)
class FinalAction:
    symbol: str
    action: str                 # 매도검토|비중축소|보류|매수후보|관찰|홀딩
    urgency: int                # _ACTION_PRIORITY 기반
    rationale: str              # 근거 + 상충 내역
    sources: tuple[str, ...] = field(default_factory=tuple)   # 관여 엔진들
    conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sources"] = list(self.sources)
        return d


def _hit_rate(calibration: dict | None, engine: str, kind: str) -> float | None:
    """n>=10 검증된 적중률만. 미달/부재 → None(미검증)."""
    if not calibration:
        return None
    entry = (calibration.get(engine) or {}).get(kind) or {}
    n = entry.get("n") or 0
    if n < _MIN_SAMPLES:
        return None
    return entry.get("hit_rate")


def _best_rate(group: list[dict], calibration: dict | None) -> float | None:
    rates = [r for r in (_hit_rate(calibration, s["engine"], s["kind"]) for s in group)
             if r is not None]
    return max(rates) if rates else None


def _mk(symbol: str, action: str, rationale: str,
        sources: list[str], conflict: bool) -> FinalAction:
    return FinalAction(symbol=symbol, action=action,
                       urgency=_ACTION_PRIORITY[action], rationale=rationale,
                       sources=tuple(sorted(set(sources))), conflict=conflict)


def _note_of(group: list[dict]) -> str:
    notes = [s.get("note", "") for s in group if s.get("note")]
    return notes[0] if notes else ""


def arbitrate(signals: list[dict[str, Any]],
              calibration: dict | None = None) -> list[FinalAction]:
    """normalize된 신호 → 종목당 단일 FinalAction. 우선순위(긴급도) 내림차순."""
    by_sym: dict[str, list[dict]] = {}
    for s in signals:
        sym = str(s.get("symbol", ""))
        if not sym:
            continue  # 매크로(EVENT 등)는 종목 중재 대상 아님
        by_sym.setdefault(sym, []).append(s)

    out: list[FinalAction] = []
    for sym, group in by_sym.items():
        engines = [s["engine"] for s in group]
        # VELOCITY<70 = 추세 경고(긴급 손절 아님)이라 비방어. ※ VELOCITY(하락장)와
        # RS_REVERT(비하락장 buy)는 레짐상 상호배타라 동시 출현 불가 — 충돌 무관.
        defensive = [s for s in group if s["intent"] == "defensive"
                     or (s["kind"] == "VELOCITY" and (s.get("urgency") or 0) >= _VELOCITY_DEFENSIVE_URGENCY)]
        buys = [s for s in group if s["intent"] == "buy"]
        trims = [s for s in group if s["intent"] == "trim"]
        warns = [s for s in group if s["intent"] == "warn" and s not in defensive]

        if defensive:
            why = _note_of(defensive) or "방어 신호 발동"
            conflict = bool(buys)
            if conflict:
                why += f" — 매수 신호({buys[0]['engine']}) 상충, 안전 우선"
            out.append(_mk(sym, "매도검토", why, engines, conflict))
            continue
        if trims:
            out.append(_mk(sym, "비중축소", _note_of(trims) or "익절/축소 신호", engines, False))
            continue
        if buys and warns:
            buy_rate = _best_rate(buys, calibration)
            warn_rate = _best_rate(warns, calibration)
            if buy_rate is not None and buy_rate > 0.0 and (warn_rate is None or buy_rate > warn_rate):
                why = (f"매수·약세 상충 — 적중률 우세({buys[0]['engine']} "
                       f"{buy_rate:.0%} vs {'미검증' if warn_rate is None else f'{warn_rate:.0%}'})")
                out.append(_mk(sym, "매수후보", why, engines, True))
            elif warn_rate is not None and (buy_rate is None or warn_rate >= buy_rate):
                why = (f"매수·약세 상충 — 적중률 우세({warns[0]['engine']} "
                       f"{warn_rate:.0%}) 약세 측")
                out.append(_mk(sym, "관찰", why, engines, True))
            else:
                out.append(_mk(sym, "보류",
                               "매수·약세 상충 — 양쪽 모두 적중률 미검증(n<10), 보수 유지",
                               engines, True))
            continue
        if buys:
            out.append(_mk(sym, "매수후보", _note_of(buys) or "매수 신호", engines, False))
            continue
        if warns:
            out.append(_mk(sym, "관찰", _note_of(warns) or "약세 경고", engines, False))
            continue
        out.append(_mk(sym, "홀딩", _note_of(group) or "신호 정상", engines, False))

    return sorted(out, key=lambda a: -a.urgency)
