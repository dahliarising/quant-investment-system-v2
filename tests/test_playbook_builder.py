"""Tests for corvin_jarvis.playbook.builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corvin_jarvis.playbook import builder


@pytest.mark.unit
def test_load_holdings_maps_symbol_to_pnl(tmp_path: Path) -> None:
    f = tmp_path / "portfolio.json"
    f.write_text(json.dumps({"holdings": [
        {"symbol": "MSFT", "shares": 6, "pnlPct": 20.2},
        {"symbol": "012450", "shares": 4, "pnlPct": -7.3},
    ]}))
    h = builder.load_holdings(f)
    assert h["MSFT"]["pnl_pct"] == 20.2
    assert h["012450"]["pnl_pct"] == -7.3


@pytest.mark.unit
def test_build_playbooks_watched_is_enter() -> None:
    universe = [{"symbol": "BWXT", "market": "US", "name": "BWX"}]
    # downward-then-flat closes → price below MAs
    closes = [200.0] * 30 + [190.0] * 30
    pbs = builder.build_playbooks(
        universe, holdings={}, fetch_prices=lambda s, m: closes
    )
    assert len(pbs) == 1
    assert pbs[0].stance == "ENTER"
    assert pbs[0].zones[0].kind == "buy"


@pytest.mark.unit
def test_build_playbooks_held_profit_is_harvest() -> None:
    universe = [{"symbol": "MSFT", "market": "US", "name": "Microsoft"}]
    closes = list(range(100, 200))  # rising → in profit, overbought
    pbs = builder.build_playbooks(
        universe, holdings={"MSFT": {"pnl_pct": 20.2}},
        fetch_prices=lambda s, m: closes,
    )
    assert pbs[0].stance == "HARVEST"
    assert pbs[0].pnl_pct == 20.2
    assert pbs[0].zones[0].kind == "trim"


@pytest.mark.unit
def test_build_playbooks_skips_insufficient_history() -> None:
    universe = [{"symbol": "X", "market": "US", "name": "X"}]
    pbs = builder.build_playbooks(
        universe, holdings={}, fetch_prices=lambda s, m: [1.0, 2.0]
    )
    assert pbs == []


@pytest.mark.unit
def test_build_playbooks_skips_on_fetch_error() -> None:
    universe = [{"symbol": "X", "market": "US", "name": "X"}]

    def boom(s: str, m: str) -> list[float]:
        raise RuntimeError("network")

    pbs = builder.build_playbooks(universe, holdings={}, fetch_prices=boom)
    assert pbs == []
