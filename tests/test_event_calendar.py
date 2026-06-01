"""Tests for corvin_jarvis.signals.event_calendar."""
from __future__ import annotations

from datetime import date

import pytest

from corvin_jarvis.signals import event_calendar


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
