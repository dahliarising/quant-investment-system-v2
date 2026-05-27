from datetime import datetime
from zoneinfo import ZoneInfo

from corvin_jarvis.signals import market_phase

KST = ZoneInfo("Asia/Seoul")


def test_kr_open_is_provisional():
    now = datetime(2026, 5, 27, 9, 13, tzinfo=KST)  # KR 장중
    assert market_phase.phase_for("KR", now) == "provisional"


def test_kr_after_close_is_confirmed():
    now = datetime(2026, 5, 27, 16, 0, tzinfo=KST)  # KR 마감 후
    assert market_phase.phase_for("KR", now) == "confirmed"


def test_us_closed_during_kr_morning_is_confirmed():
    now = datetime(2026, 5, 27, 9, 13, tzinfo=KST)  # 美장 마감 상태
    assert market_phase.phase_for("US", now) == "confirmed"


def test_unknown_market_defaults_confirmed():
    now = datetime(2026, 5, 27, 9, 13, tzinfo=KST)
    assert market_phase.phase_for("XX", now) == "confirmed"


def test_kr_exactly_at_close_is_confirmed():
    now = datetime(2026, 5, 27, 15, 30, tzinfo=KST)  # 종가 시점 = 완성봉
    assert market_phase.phase_for("KR", now) == "confirmed"


def test_kr_weekend_is_confirmed():
    now = datetime(2026, 5, 30, 10, 0, tzinfo=KST)  # 토요일
    assert market_phase.phase_for("KR", now) == "confirmed"


def test_us_during_session_is_provisional():
    now = datetime(2026, 5, 27, 14, 0, tzinfo=ZoneInfo("America/New_York"))  # 美 장중
    assert market_phase.phase_for("US", now) == "provisional"


def test_naive_datetime_raises():
    import pytest
    with pytest.raises(ValueError):
        market_phase.phase_for("KR", datetime(2026, 5, 27, 9, 13))  # tz 없음
