import json
from pathlib import Path
from corvin_jarvis.dashboard import equity_curve


def _write(p: Path, name: str, data: dict):
    (p / name).write_text(json.dumps(data), encoding="utf-8")


def test_extract_total_prefers_total_assets():
    assert equity_curve._extract_total({"totals": {"totalAssetsKRW": 100, "valueKRW": 50}}) == 100


def test_extract_total_falls_back_to_value_krw():
    assert equity_curve._extract_total({"totals": {"valueKRW": 50}}) == 50


def test_extract_total_falls_back_to_holdings_sum():
    data = {"holdings": [{"valueKRW": 30}, {"valueKRW": 20}]}
    assert equity_curve._extract_total(data) == 50


def test_extract_total_returns_none_when_unusable():
    assert equity_curve._extract_total({"foo": "bar"}) is None


def test_build_series_sorted_and_skips_bad(tmp_path):
    _write(tmp_path, "portfolio.json.bak.2026-05-19", {"totals": {"totalAssetsKRW": 100}})
    _write(tmp_path, "portfolio.json.bak.2026-05-28T1419", {"totals": {"totalAssetsKRW": 120}})
    _write(tmp_path, "portfolio.json.bak.baddate", {"totals": {"totalAssetsKRW": 999}})
    _write(tmp_path, "portfolio.json", {"totals": {"totalAssetsKRW": 130}})
    series = equity_curve.build_series(tmp_path)
    assert [p["value"] for p in series] == [100, 120, 130]
    assert series[0]["date"] == "2026-05-19"
    assert series[-1]["value"] == 130
