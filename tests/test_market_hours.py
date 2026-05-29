"""Tests for corvin_jarvis.market_hours — 시장 영업시간 기반 alert 억제."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from corvin_jarvis import market_hours as mh

KST = ZoneInfo("Asia/Seoul")

# 2026-05-29 = 금요일
FRI_KR_OPEN = datetime(2026, 5, 29, 11, 0, tzinfo=KST)    # KR 장중, US 휴장
FRI_KR_PREOPEN = datetime(2026, 5, 29, 8, 0, tzinfo=KST)  # KR 개장 전, US 휴장
FRI_US_OPEN = datetime(2026, 5, 29, 23, 0, tzinfo=KST)    # KR 휴장, US 장중(10:00 ET)
SAT = datetime(2026, 5, 30, 11, 0, tzinfo=KST)            # 토요일 양 시장 휴장

SECTOR_MKT = {
    "battery": {"KR"}, "space": {"US"}, "semiconductor": {"KR", "US"}, "auto": {"KR", "US"},
}


def test_kr_open_hours():
    assert mh.is_kr_open(FRI_KR_OPEN) is True
    assert mh.is_kr_open(FRI_KR_PREOPEN) is False   # 09:00 전
    assert mh.is_kr_open(datetime(2026, 5, 29, 15, 31, tzinfo=KST)) is False  # 15:30 후
    assert mh.is_kr_open(SAT) is False


def test_us_open_hours():
    assert mh.is_us_open(FRI_US_OPEN) is True        # 10:00 ET
    assert mh.is_us_open(FRI_KR_OPEN) is False        # 11:00 KST = 전일 22:00 ET
    assert mh.is_us_open(SAT) is False


def test_alert_markets_classification():
    assert mh.alert_markets({"category": "universe", "metric": "universe_LUNR_confirmed"}, SECTOR_MKT) == {"US"}
    assert mh.alert_markets({"category": "universe", "metric": "universe_005930_provisional"}, SECTOR_MKT) == {"KR"}
    assert mh.alert_markets({"category": "leading_rs", "metric": "rs_005380_provisional"}, SECTOR_MKT) == {"KR"}
    assert mh.alert_markets({"category": "portfolio", "metric": "pnl_MSFT"}, SECTOR_MKT) == {"US"}
    assert mh.alert_markets({"category": "index", "metric": "kospi"}, SECTOR_MKT) == {"KR"}
    assert mh.alert_markets({"category": "index", "metric": "nasdaq"}, SECTOR_MKT) == {"US"}
    assert mh.alert_markets({"category": "sector", "metric": "sector_space_confirmed"}, SECTOR_MKT) == {"US"}
    assert mh.alert_markets({"category": "sector", "metric": "sector_semiconductor_provisional"}, SECTOR_MKT) == {"KR", "US"}
    assert mh.alert_markets({"category": "fx", "metric": "usd_krw_level"}, SECTOR_MKT) == set()


def test_suppress_us_stock_when_us_closed():
    # 핵심: LUNR(US) alert이 KST 낮(US 휴장)에 억제됨
    a = {"category": "universe", "metric": "universe_LUNR_confirmed"}
    assert mh.should_suppress(a, FRI_KR_OPEN, SECTOR_MKT) is True


def test_keep_kr_stock_when_kr_open():
    a = {"category": "universe", "metric": "universe_005930_provisional"}
    assert mh.should_suppress(a, FRI_KR_OPEN, SECTOR_MKT) is False


def test_macro_always_sent():
    a = {"category": "fx", "metric": "usd_krw_level"}
    assert mh.should_suppress(a, FRI_KR_OPEN, SECTOR_MKT) is False
    assert mh.should_suppress(a, SAT, SECTOR_MKT) is False


def test_mixed_sector_suppressed_only_when_all_closed():
    a = {"category": "sector", "metric": "sector_semiconductor_confirmed"}  # {KR,US}
    assert mh.should_suppress(a, FRI_KR_OPEN, SECTOR_MKT) is False  # KR 열림 → 유지
    assert mh.should_suppress(a, SAT, SECTOR_MKT) is True           # 양 시장 휴장 → 억제


def test_us_stock_kept_when_us_open():
    a = {"category": "universe", "metric": "universe_LUNR_confirmed"}
    assert mh.should_suppress(a, FRI_US_OPEN, SECTOR_MKT) is False


def test_load_sector_markets_reads_file():
    m = mh.load_sector_markets()
    # 실제 monitored_universe.json 기준 — battery=KR, space=US, semiconductor 혼합
    assert m.get("battery") == {"KR"}
    assert m.get("space") == {"US"}
    assert m.get("semiconductor") == {"KR", "US"}
