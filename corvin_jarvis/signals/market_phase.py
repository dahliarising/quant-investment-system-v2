"""감시 종목의 phase 판정: 시장이 열려 있으면 provisional(미완성봉), 닫혀 있으면 confirmed."""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
NY = ZoneInfo("America/New_York")

# 정규장 시간 (장중 = provisional)
_KR_OPEN, _KR_CLOSE = time(9, 0), time(15, 30)
_US_OPEN, _US_CLOSE = time(9, 30), time(16, 0)


def _is_session_open(local_dt: datetime, open_t: time, close_t: time) -> bool:
    if local_dt.weekday() >= 5:  # 토(5)/일(6)
        return False
    return open_t <= local_dt.time() < close_t


def phase_for(market: str, now: datetime | None = None) -> str:
    """market 정규장이 열려 있으면 'provisional', 아니면 'confirmed'."""
    now = now or datetime.now(KST)
    if now.tzinfo is None:
        raise ValueError(f"phase_for requires a timezone-aware datetime, got naive: {now!r}")
    if market == "KR":
        local = now.astimezone(KST)
        return "provisional" if _is_session_open(local, _KR_OPEN, _KR_CLOSE) else "confirmed"
    if market == "US":
        local = now.astimezone(NY)
        return "provisional" if _is_session_open(local, _US_OPEN, _US_CLOSE) else "confirmed"
    return "confirmed"
