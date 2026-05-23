"""Tests for corvin_jarvis.attribution — outcome attribution."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from corvin_jarvis import attribution, timeseries


@pytest.mark.unit
def test_session_symbols_extracts_us_tickers() -> None:
    text = """
    오늘 META 관망. MSFT 보유 유지. NVDA 분할매수 zone.
    KOSPI 7200 지지선. TSLA 신규 검토.
    """
    syms = attribution.session_symbols(text)
    assert "META" in syms
    assert "MSFT" in syms
    assert "NVDA" in syms
    assert "TSLA" in syms


@pytest.mark.unit
def test_session_symbols_extracts_korean_codes() -> None:
    text = "삼성전자(005930) 매도 완료. 하이닉스(000660) 신규 진입."
    syms = attribution.session_symbols(text)
    assert "005930" in syms
    assert "000660" in syms


@pytest.mark.unit
def test_session_symbols_filters_noise_words() -> None:
    """공통 영어 단어가 ticker로 잘못 인식되지 않아야."""
    text = "오늘은 RISK가 높다. THE 시장이 ABOUT to crash. AT 1500 KRW."
    syms = attribution.session_symbols(text)
    # 영어 단어 중 stop-list에 있는 건 제외
    assert "THE" not in syms
    assert "AT" not in syms
    assert "ABOUT" not in syms


@pytest.mark.unit
def test_actual_moves_uses_timeseries(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    # 2 snapshots — old (start) + new (end)
    for sym, prices in [("META", [600.0, 612.0]), ("MSFT", [400.0, 396.0])]:
        for i, p in enumerate(prices):
            snap = {
                "timestamp_utc": f"2026-05-{15+i*7:02d}T00:00:00+00:00",
                "timestamp_kst": f"2026-05-{15+i*7:02d}T09:00:00+09:00",
                "indices": {},
                "commodities": {}, "fx": {}, "watchlist": [],
                "portfolio": [{
                    "symbol": sym, "shares": 1, "avg_price": p, "currency": "USD",
                    "current_price": p, "pnl_pct": 0.0, "market_value": p, "error": None,
                }],
            }
            timeseries.write_snapshot(tmp_db_path, snap)
    moves = attribution.actual_moves(["META", "MSFT"], tmp_db_path)
    by_sym = {m["symbol"]: m for m in moves}
    assert by_sym["META"]["pct_change"] == pytest.approx(2.0, abs=0.01)
    assert by_sym["MSFT"]["pct_change"] == pytest.approx(-1.0, abs=0.01)


@pytest.mark.unit
def test_actual_moves_handles_missing_symbol(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    moves = attribution.actual_moves(["XYZ"], tmp_db_path)
    assert moves == [{"symbol": "XYZ", "pct_change": None, "n": 0}]


@pytest.mark.unit
def test_weekly_report_aggregates_sessions(tmp_db_path: Path, tmp_path: Path) -> None:
    wiki = tmp_path / "corvin-sessions"
    wiki.mkdir()
    (wiki / "2026-05-19-strategy.md").write_text(
        "오늘 META 보유 유지. MSFT 분할매수.\n"
    )
    (wiki / "2026-05-20-strategy.md").write_text(
        "NVDA 분할매도 권고. KOSPI 관망.\n"
    )
    timeseries.init_db(tmp_db_path)
    for sym in ["META", "MSFT", "NVDA"]:
        for i, p in enumerate([100.0, 110.0]):
            snap = {
                "timestamp_utc": f"2026-05-{15+i*7:02d}T00:00:00+00:00",
                "timestamp_kst": f"2026-05-{15+i*7:02d}T09:00:00+09:00",
                "indices": {}, "commodities": {}, "fx": {}, "watchlist": [],
                "portfolio": [{
                    "symbol": sym, "shares": 1, "avg_price": p, "currency": "USD",
                    "current_price": p, "pnl_pct": 0.0, "market_value": p, "error": None,
                }],
            }
            timeseries.write_snapshot(tmp_db_path, snap)
    report = attribution.weekly_report(
        wiki, tmp_db_path, today=date(2026, 5, 22), lookback_days=7,
    )
    assert report["sessions_analyzed"] == 2
    assert len(report["entries"]) == 2
    # entries 안에 symbol 별 move 들어있어야
    all_syms = {sym for e in report["entries"] for sym in e["symbols"]}
    assert "META" in all_syms or "MSFT" in all_syms


@pytest.mark.unit
def test_weekly_report_handles_empty_wiki(tmp_db_path: Path, tmp_path: Path) -> None:
    wiki = tmp_path / "empty"
    timeseries.init_db(tmp_db_path)
    report = attribution.weekly_report(
        wiki, tmp_db_path, today=date(2026, 5, 22), lookback_days=7,
    )
    assert report["sessions_analyzed"] == 0
    assert report["entries"] == []
