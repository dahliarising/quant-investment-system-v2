"""Tests for corvin_jarvis.qa Q&A retrieval toolkit."""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis import qa, timeseries


def _seed_history(db_path: Path, symbol: str, prices: list[float], category: str = "portfolio") -> None:
    """헬퍼: timeseries.db에 N일치 가짜 가격 row 삽입."""
    timeseries.init_db(db_path)
    for i, p in enumerate(prices):
        ts = f"2026-05-{15+i:02d}T00:00:00+00:00"
        kst = f"2026-05-{15+i:02d}T09:00:00+09:00"
        snap = {
            "timestamp_utc": ts,
            "timestamp_kst": kst,
            "indices": {},
            "commodities": {},
            "fx": {},
            "portfolio": [],
            "watchlist": [],
        }
        if category == "portfolio":
            snap["portfolio"] = [{
                "symbol": symbol, "shares": 1, "avg_price": p, "currency": "USD",
                "current_price": p, "pnl_pct": 0.0, "market_value": p, "error": None,
            }]
        elif category == "index":
            snap["indices"] = {symbol: {"price": p, "pct_change": 0.0, "source": "test", "error": None}}
        timeseries.write_snapshot(db_path, snap)


@pytest.mark.unit
def test_recent_history_summary_basic(tmp_db_path: Path) -> None:
    _seed_history(tmp_db_path, "META", [600.0, 605.0, 595.0, 610.0, 605.0])
    summary = qa.recent_history_summary(tmp_db_path, symbol="META", days=10)
    assert summary["symbol"] == "META"
    assert summary["n"] == 5
    assert summary["start_price"] == 600.0
    assert summary["end_price"] == 605.0
    assert summary["min_price"] == 595.0
    assert summary["max_price"] == 610.0
    assert summary["pct_change"] == pytest.approx((605 / 600 - 1) * 100, abs=0.01)


@pytest.mark.unit
def test_recent_history_summary_missing_symbol(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    summary = qa.recent_history_summary(tmp_db_path, symbol="UNKNOWN", days=10)
    assert summary["n"] == 0
    assert summary["start_price"] is None


@pytest.mark.unit
def test_relative_strength_outperform(tmp_db_path: Path) -> None:
    """META +5%, SP500 +2% → RS = +3pp"""
    _seed_history(tmp_db_path, "META", [600.0, 610.0, 615.0, 620.0, 630.0])
    _seed_history(tmp_db_path, "sp500", [5000.0, 5050.0, 5070.0, 5080.0, 5100.0], category="index")
    rs = qa.relative_strength(tmp_db_path, symbol="META", benchmark="sp500", days=10)
    assert rs["symbol_pct"] == pytest.approx(5.0, abs=0.01)
    assert rs["benchmark_pct"] == pytest.approx(2.0, abs=0.01)
    assert rs["relative_pp"] == pytest.approx(3.0, abs=0.01)
    assert rs["verdict"] == "outperform"


@pytest.mark.unit
def test_relative_strength_underperform(tmp_db_path: Path) -> None:
    """META -2%, SP500 +1% → RS = -3pp"""
    _seed_history(tmp_db_path, "META", [600.0, 595.0, 590.0, 595.0, 588.0])
    _seed_history(tmp_db_path, "sp500", [5000.0, 5025.0, 5040.0, 5050.0, 5050.0], category="index")
    rs = qa.relative_strength(tmp_db_path, symbol="META", benchmark="sp500", days=10)
    assert rs["verdict"] == "underperform"
    assert rs["relative_pp"] < 0


@pytest.mark.unit
def test_relative_strength_handles_missing_data(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    rs = qa.relative_strength(tmp_db_path, symbol="X", benchmark="Y", days=10)
    assert rs["verdict"] == "unknown"
    assert rs["symbol_pct"] is None


@pytest.mark.unit
def test_wiki_search_finds_matching_files(tmp_path: Path) -> None:
    wiki = tmp_path / "corvin-sessions"
    wiki.mkdir()
    (wiki / "2026-05-19-portfolio-resync.md").write_text(
        "META 7주 보유. 5/19 portfolio sync. avgPrice 597.61\n"
    )
    (wiki / "2026-05-20-strategy.md").write_text(
        "오늘 KOSPI 7200 하락. META 관망 zone.\n"
    )
    (wiki / "2026-04-22-old.md").write_text("샘성전자 매도 완료\n")

    hits = qa.wiki_search("META", wiki_dir=wiki, top_k=5)
    assert len(hits) == 2
    paths = {h["path"].name for h in hits}
    assert "2026-05-19-portfolio-resync.md" in paths
    assert "2026-05-20-strategy.md" in paths
    assert all("META" in h["snippet"] for h in hits)


@pytest.mark.unit
def test_wiki_search_respects_top_k(tmp_path: Path) -> None:
    wiki = tmp_path / "corvin-sessions"
    wiki.mkdir()
    for i in range(5):
        (wiki / f"2026-05-{20+i}-x.md").write_text(f"META mention {i}\n")
    hits = qa.wiki_search("META", wiki_dir=wiki, top_k=2)
    assert len(hits) == 2


@pytest.mark.unit
def test_wiki_search_returns_empty_when_no_match(tmp_path: Path) -> None:
    wiki = tmp_path / "corvin-sessions"
    wiki.mkdir()
    (wiki / "a.md").write_text("nothing here\n")
    hits = qa.wiki_search("META", wiki_dir=wiki, top_k=5)
    assert hits == []


@pytest.mark.unit
def test_wiki_search_handles_missing_dir(tmp_path: Path) -> None:
    hits = qa.wiki_search("META", wiki_dir=tmp_path / "nonexistent", top_k=5)
    assert hits == []


@pytest.mark.unit
def test_explain_move_combines_sources(tmp_db_path: Path, tmp_path: Path) -> None:
    """explain_move는 history + relative + wiki를 묶어 단일 dict 반환."""
    _seed_history(tmp_db_path, "META", [600.0, 595.0, 590.0, 595.0, 588.0])
    _seed_history(tmp_db_path, "sp500", [5000.0, 5025.0, 5040.0, 5050.0, 5050.0], category="index")

    wiki = tmp_path / "corvin-sessions"
    wiki.mkdir()
    (wiki / "2026-05-20-meta.md").write_text("META 관망 zone, 590 지지선\n")

    out = qa.explain_move(
        tmp_db_path, symbol="META", days=7,
        benchmark="sp500", wiki_dir=wiki,
    )
    assert out["symbol"] == "META"
    assert out["history"]["n"] == 5
    assert out["relative"]["verdict"] == "underperform"
    assert len(out["wiki_hits"]) == 1
    assert "META" in out["wiki_hits"][0]["snippet"]


@pytest.mark.unit
def test_explain_move_handles_no_data_gracefully(tmp_db_path: Path, tmp_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    out = qa.explain_move(
        tmp_db_path, symbol="NEW", days=7,
        benchmark="sp500", wiki_dir=tmp_path / "empty",
    )
    assert out["history"]["n"] == 0
    assert out["relative"]["verdict"] == "unknown"
    assert out["wiki_hits"] == []
