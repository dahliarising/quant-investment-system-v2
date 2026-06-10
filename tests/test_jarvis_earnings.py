"""Tests for jarvis earnings integration."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_collect_us_symbols_skips_korean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({"holdings": [
        {"symbol": "META", "shares": 1},
        {"symbol": "005930", "shares": 10},
        {"symbol": "MSFT", "shares": 2},
    ]}))
    cfg_dir = tmp_path / "cjarvis"
    cfg_dir.mkdir()
    cfg = cfg_dir / "config.json"
    cfg.write_text(json.dumps({"watchlist": ["TSLA", "000660", "AAPL.KS"]}))

    from corvin_jarvis import jarvis
    monkeypatch.setattr(jarvis, "BASE_DIR", cfg_dir)

    syms = jarvis._collect_us_symbols()
    assert "META" in syms
    assert "MSFT" in syms
    assert "TSLA" in syms
    assert "005930" not in syms
    assert "000660" not in syms
    assert "AAPL.KS" not in syms


@pytest.mark.unit
def test_alert_kind_symbol_split() -> None:
    from corvin_jarvis.jarvis import _alert_kind_symbol
    assert _alert_kind_symbol("rs_NVDA_confirmed", "rs") == ("rs_confirmed", "NVDA")
    assert _alert_kind_symbol("stop_loss_012450", "portfolio") == ("stop_loss", "012450")
    assert _alert_kind_symbol("kospi", "index") == ("kospi", "")
    assert _alert_kind_symbol("META", "portfolio") == ("portfolio", "META")
    assert _alert_kind_symbol("sector_auto_KR_provisional", "sector") == ("sector_auto_KR_provisional", "")
    assert _alert_kind_symbol("usd_krw_level", "fx") == ("usd_krw_level", "")
