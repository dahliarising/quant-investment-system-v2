"""Tests for corvin_jarvis.signals.event_calendar."""
from __future__ import annotations

from datetime import date

import pytest

from corvin_jarvis.signals import event_calendar
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_macro_events_returns_known_dates():
    # 하드코딩 거시 캘린더에서 FOMC/BOK 일정 조회
    events = event_calendar.macro_events_within(
        as_of=date(2026, 6, 1), horizon_days=30
    )
    # 각 이벤트는 name, event_date 보유
    assert all("name" in e and "event_date" in e for e in events)
    # 30일 내 이벤트만
    for e in events:
        delta = (e["event_date"] - date(2026, 6, 1)).days
        assert 0 <= delta <= 30


@pytest.mark.unit
def test_macro_events_excludes_past():
    events = event_calendar.macro_events_within(
        as_of=date(2026, 6, 1), horizon_days=30
    )
    for e in events:
        assert e["event_date"] >= date(2026, 6, 1)


@pytest.mark.unit
def test_build_event_signals_earnings_dn():
    # 실적이 D-3이면 event 신호 생성
    earnings = [{"symbol": "012450", "earnings_date": date(2026, 6, 4)}]
    sigs = event_calendar.build_event_signals(
        as_of=date(2026, 6, 1),
        earnings_rows=earnings,
        macro_horizon_days=30,
    )
    earn_sigs = [s for s in sigs if s.symbol == "012450"]
    assert len(earn_sigs) == 1
    s = earn_sigs[0]
    assert isinstance(s, LeadingSignal)
    assert s.pillar == "event"
    assert s.direction == "neutral"      # 이벤트는 방향성 없음
    assert s.evidence["days_to"] == 3
    assert "D-3" in s.message


@pytest.mark.unit
def test_build_event_signals_skips_far_earnings():
    # D-10은 ALERT_OFFSETS(7,3,1) 밖 → 신호 없음
    earnings = [{"symbol": "012450", "earnings_date": date(2026, 6, 11)}]
    sigs = event_calendar.build_event_signals(
        as_of=date(2026, 6, 1), earnings_rows=earnings, macro_horizon_days=0
    )
    assert [s for s in sigs if s.symbol == "012450"] == []


@pytest.mark.unit
def test_build_event_signals_macro_advisory():
    # 거시 이벤트는 symbol="_MACRO"로 시장 전체 대상, 확정 신뢰도
    sigs = event_calendar.build_event_signals(
        as_of=date(2026, 6, 1), earnings_rows=[], macro_horizon_days=30
    )
    macro = [s for s in sigs if s.pillar == "event" and s.symbol == "_MACRO"]
    assert len(macro) >= 1
    assert all(s.confidence >= 60.0 for s in macro)


@pytest.mark.unit
def test_filter_dart_keeps_catalyst_keywords():
    disclosures = [
        {"report_nm": "단일판매ㆍ공급계약체결", "rcept_dt": "20260601"},
        {"report_nm": "주주총회소집결의", "rcept_dt": "20260601"},  # 비촉매
        {"report_nm": "유상증자결정", "rcept_dt": "20260601"},
    ]
    kept = event_calendar.filter_dart_disclosures(disclosures)
    names = [d["report_nm"] for d in kept]
    assert "단일판매ㆍ공급계약체결" in names
    assert "유상증자결정" in names
    assert "주주총회소집결의" not in names


@pytest.mark.unit
def test_build_dart_signals():
    disclosures = [{"report_nm": "단일판매ㆍ공급계약체결", "rcept_dt": "20260601"}]
    sigs = event_calendar.build_dart_signals("012450", disclosures)
    assert len(sigs) == 1
    s = sigs[0]
    assert s.pillar == "event"
    assert s.symbol == "012450"
    assert s.confidence >= 60.0
    assert "공급계약" in s.message or "계약" in s.message


@pytest.mark.unit
def test_build_dart_signals_empty():
    assert event_calendar.build_dart_signals("012450", []) == []


@pytest.mark.unit
def test_bok_calendar_entries_in_valid_decision_months():
    """재발 방지: BOK 항목은 실제 금리결정 달(1·2·4·5·7·8·10·11)에만.

    2026-06-11 오발주 버그 회귀 가드 — 6·9·12월은 금융안정회의(금리 무관).
    """
    from datetime import date as _date
    for name, iso in event_calendar.MACRO_CALENDAR:
        if "BOK" in name:
            m = _date.fromisoformat(iso).month
            assert m in event_calendar._BOK_DECISION_MONTHS, \
                f"BOK {iso}: 월 {m}은 금리결정 달이 아님"


@pytest.mark.unit
def test_no_bok_event_in_june_2026():
    """6/11 가짜 BOK 경보 회귀 가드 — 6월엔 BOK 금리 이벤트 없음."""
    events = event_calendar.macro_events_within(as_of=date(2026, 6, 1), horizon_days=20)
    assert not any("BOK" in e["name"] for e in events)
