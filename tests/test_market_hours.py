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


def test_sector_alert_with_explicit_market_field_is_market_precise():
    # ⑤ 시장별 분리 후: market 필드가 있으면 그 시장만 보고 억제 판정
    kr = {"category": "sector", "metric": "sector_semiconductor_KR_confirmed", "market": "KR"}
    us = {"category": "sector", "metric": "sector_semiconductor_US_confirmed", "market": "US"}
    assert mh.alert_markets(kr, SECTOR_MKT) == {"KR"}
    assert mh.alert_markets(us, SECTOR_MKT) == {"US"}
    # KR 장중·US 휴장: KR 섹터는 유지, US 섹터는 억제 (혼합이었으면 둘 다 유지됐음)
    assert mh.should_suppress(kr, FRI_KR_OPEN, SECTOR_MKT) is False
    assert mh.should_suppress(us, FRI_KR_OPEN, SECTOR_MKT) is True


def test_us_stock_kept_when_us_open():
    a = {"category": "universe", "metric": "universe_LUNR_confirmed"}
    assert mh.should_suppress(a, FRI_US_OPEN, SECTOR_MKT) is False


def test_load_sector_markets_reads_file():
    m = mh.load_sector_markets()
    # 실제 monitored_universe.json 기준 — battery=KR, space=US, semiconductor 혼합
    assert m.get("battery") == {"KR"}
    assert m.get("space") == {"US"}
    assert m.get("semiconductor") == {"KR", "US"}


# ---------------------------------------------------------------------------
# 2026-06-14 (일): 주말 브리핑이 금요일 종가를 '현재가'로 찍어 '예전 그대로'로
# 보이던 문제. predictive 카테고리 휴장 우회 버그(C) + 라벨/억제 헬퍼.
# ---------------------------------------------------------------------------
from datetime import date  # noqa: E402

SUN_0500 = datetime(2026, 6, 14, 5, 0, tzinfo=KST)   # 일요일 새벽 (브리핑 시각)
FRI_1000 = datetime(2026, 6, 12, 10, 0, tzinfo=KST)  # 금요일 KR 정규장 중
SAT_1200 = datetime(2026, 6, 13, 12, 0, tzinfo=KST)  # 토요일


def test_predictive_alert_maps_to_symbol_market():
    """C: predictive 알람은 종목 시장으로 매핑돼야 한다 (거시 취급 금지)."""
    assert mh.alert_markets({"category": "predictive", "metric": "META"}, {}) == {"US"}
    assert mh.alert_markets({"category": "predictive", "metric": "005930"}, {}) == {"KR"}


def test_predictive_suppressed_on_weekend():
    """C: 주말엔 predictive 종목 알람도 억제 대상."""
    assert mh.should_suppress({"category": "predictive", "metric": "META"}, SUN_0500, {}) is True


def test_early_warning_hardstop_maps_to_symbol_market():
    """early_warning 하드스톱도 종목 시장으로 매핑 (거시 우회 금지)."""
    assert mh.alert_markets({"category": "early_warning", "metric": "hardstop_012450"}, {}) == {"KR"}
    assert mh.alert_markets({"category": "early_warning", "metric": "hardstop_NVDA"}, {}) == {"US"}


def test_early_warning_suppressed_on_weekend():
    """주말엔 하드스톱 긴급알람도 억제 (월요일 개장 전 재부상)."""
    a = {"category": "early_warning", "metric": "hardstop_012450"}
    assert mh.should_suppress(a, SUN_0500, {}) is True
    # 평일 KR 장중엔 유지
    assert mh.should_suppress(a, FRI_1000, {}) is False


def test_last_close_date_weekend_is_friday():
    assert mh.last_close_date(SUN_0500, "KR") == date(2026, 6, 12)
    # 일요일 05:00 KST = 토요일 16:00 ET → US 마지막 마감도 금요일
    assert mh.last_close_date(SUN_0500, "US") == date(2026, 6, 12)


def test_last_close_date_intraday_not_yet_closed_uses_prev():
    # 금요일 10:00 KST — 아직 장중(마감 전) → 마지막 '완료' 거래일은 목요일
    assert mh.last_close_date(FRI_1000, "KR") == date(2026, 6, 11)


def test_next_open_date_weekend_is_monday():
    assert mh.next_open_date(SUN_0500, "KR") == date(2026, 6, 15)
    assert mh.next_open_date(SAT_1200, "KR") == date(2026, 6, 15)


def test_market_status_label_weekend_flags_closed_and_vintage():
    label = mh.market_status_label(SUN_0500)
    assert "휴장" in label
    assert "06-12" in label   # 마지막 거래일 종가 기준 표기
    assert "06-15" in label   # 다음 개장


def test_market_status_label_open_session_marks_realtime():
    label = mh.market_status_label(FRI_1000)
    assert "정규장" in label or "개장" in label


def test_partition_alerts_splits_live_and_stale_on_weekend():
    alerts = [
        {"category": "fx", "metric": "usd_krw_level", "severity": "low"},       # live (macro)
        {"category": "index", "metric": "kospi", "severity": "critical"},       # stale (KR 휴장)
        {"category": "predictive", "metric": "META", "severity": "critical"},   # stale (US 휴장)
        {"category": "commodity", "metric": "gold", "severity": "medium"},      # live (macro)
    ]
    live, stale = mh.partition_alerts(alerts, SUN_0500, {})
    assert {a["metric"] for a in live} == {"usd_krw_level", "gold"}
    assert {a["metric"] for a in stale} == {"kospi", "META"}


def test_partition_alerts_all_live_during_open_session():
    alerts = [
        {"category": "index", "metric": "kospi", "severity": "critical"},
        {"category": "fx", "metric": "usd_krw_level", "severity": "low"},
    ]
    live, stale = mh.partition_alerts(alerts, FRI_1000, {})
    assert len(live) == 2
    assert stale == []
