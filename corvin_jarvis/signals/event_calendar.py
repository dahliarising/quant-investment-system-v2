"""Corvin 선행 인텔리전스 — Pillar 2: 이벤트 지평선 캘린더.

가격과 무관하게 "알려진 미래 촉매"를 미리 경보 → 진짜 시장 선행성.
- 거시 일정(FOMC/BOK): 하드코딩 캘린더 (분기 갱신)
- 실적 D-N: 기존 earnings.py 재사용 (build_event_signals)

모든 함수는 as_of(date)를 명시 인자로 받아 결정성·테스트가능성을 보장한다.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from corvin_jarvis.signals.leading_signal import LeadingSignal

# 분기마다 갱신하는 거시 캘린더. (name, ISO date)
# 2026 FOMC/BOK 일정 — 갱신 시 이 리스트만 수정.
MACRO_CALENDAR: list[tuple[str, str]] = [
    ("FOMC 금리결정", "2026-06-17"),
    ("BOK 금융통화위원회", "2026-06-11"),
    ("FOMC 금리결정", "2026-07-29"),
    ("BOK 금융통화위원회", "2026-07-09"),
]

# earnings.py와 동일한 알림 시점 (D-7/3/1)
EARNINGS_ALERT_OFFSETS = (7, 3, 1)


def macro_events_within(as_of: date, horizon_days: int = 30) -> list[dict[str, Any]]:
    """as_of 기준 horizon_days 이내의 거시 이벤트만 반환."""
    out: list[dict[str, Any]] = []
    for name, iso in MACRO_CALENDAR:
        ev = date.fromisoformat(iso)
        delta = (ev - as_of).days
        if 0 <= delta <= horizon_days:
            out.append({"name": name, "event_date": ev, "days_to": delta})
    return out


def build_event_signals(
    as_of: date,
    earnings_rows: list[dict[str, Any]],
    macro_horizon_days: int = 30,
) -> list[LeadingSignal]:
    """실적 D-N + 거시 일정 → LeadingSignal 리스트.

    earnings_rows: [{"symbol": str, "earnings_date": date}, ...] (호출부가 DB에서 조회).
    이벤트는 방향성 없음 → direction="neutral", horizon="days".
    """
    signals: list[LeadingSignal] = []

    # 실적 D-N
    for row in earnings_rows:
        sym = row["symbol"]
        ev = row["earnings_date"]
        days_to = (ev - as_of).days
        if days_to in EARNINGS_ALERT_OFFSETS:
            signals.append(LeadingSignal(
                pillar="event", symbol=sym, direction="neutral",
                confidence=80.0, score=None, horizon="days",
                advisory=False,
                message=f"📅 {sym} 실적 D-{days_to} ({ev.isoformat()})",
                evidence={"kind": "earnings", "days_to": days_to,
                          "event_date": ev.isoformat()},
            ))

    # 거시 일정 (시장 전체 대상 → symbol="_MACRO")
    for ev in macro_events_within(as_of, macro_horizon_days):
        signals.append(LeadingSignal(
            pillar="event", symbol="_MACRO", direction="neutral",
            confidence=75.0, score=None, horizon="days",
            advisory=False,
            message=f"🏛️ {ev['name']} D-{ev['days_to']} ({ev['event_date'].isoformat()})",
            evidence={"kind": "macro", "days_to": ev["days_to"],
                      "name": ev["name"]},
        ))

    return signals
