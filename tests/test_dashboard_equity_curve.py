import json
from pathlib import Path
from corvin_jarvis.dashboard import equity_curve


def _write(p: Path, name: str, data: dict):
    (p / name).write_text(json.dumps(data), encoding="utf-8")


def test_extract_pnl_pct_prefers_equity_pnl_pct():
    assert equity_curve._extract_pnl_pct({"totals": {"equityPnlPct": 1.95, "pnlPct": 7.9}}) == 1.95


def test_extract_pnl_pct_falls_back_to_pnl_pct():
    assert equity_curve._extract_pnl_pct({"totals": {"pnlPct": 7.9}}) == 7.9


def test_extract_pnl_pct_returns_none_when_absent():
    assert equity_curve._extract_pnl_pct({"totals": {"valueKRW": 100}}) is None


def test_build_series_returns_pnl_pct_points(tmp_path):
    _write(tmp_path, "portfolio.json.bak.2026-05-19", {"totals": {"pnlPct": 5.0}})
    _write(tmp_path, "portfolio.json.bak.2026-05-28", {"totals": {"pnlPct": 6.0}})
    series = equity_curve.build_series(tmp_path)
    assert series == [{"date": "2026-05-19", "pnl_pct": 5.0},
                      {"date": "2026-05-28", "pnl_pct": 6.0}]


def test_build_series_dedupes_date_keeping_latest_timestamp(tmp_path):
    # bare-date and timestamped backup on the same day -> later timestamp wins, one point only
    _write(tmp_path, "portfolio.json.bak.2026-05-28", {"totals": {"pnlPct": 6.0}})
    _write(tmp_path, "portfolio.json.bak.2026-05-28T1419", {"totals": {"pnlPct": 6.5}})
    series = equity_curve.build_series(tmp_path)
    pts = [p for p in series if p["date"] == "2026-05-28"]
    assert len(pts) == 1
    assert pts[0]["pnl_pct"] == 6.5


def test_build_series_current_portfolio_wins_for_its_date(tmp_path):
    _write(tmp_path, "portfolio.json.bak.2026-06-05T0610", {"totals": {"equityPnlPct": 1.95}})
    _write(tmp_path, "portfolio.json", {"updatedAt": "2026-06-05", "totals": {"equityPnlPct": 3.0}})
    series = equity_curve.build_series(tmp_path)
    pts = [p for p in series if p["date"] == "2026-06-05"]
    assert len(pts) == 1
    assert pts[0]["pnl_pct"] == 3.0


def test_build_series_skips_bad_dates_and_missing_pnl(tmp_path):
    _write(tmp_path, "portfolio.json.bak.baddate", {"totals": {"pnlPct": 9.9}})
    _write(tmp_path, "portfolio.json.bak.2026-05-19", {"totals": {"valueKRW": 100}})  # no pnl%
    _write(tmp_path, "portfolio.json.bak.2026-05-20", {"totals": {"pnlPct": 4.0}})
    series = equity_curve.build_series(tmp_path)
    assert series == [{"date": "2026-05-20", "pnl_pct": 4.0}]
