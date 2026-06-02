"""Tests for corvin_jarvis.playbook.render_html."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import render_html
from corvin_jarvis.playbook.models import Playbook, Technicals, Zone


def _pb(symbol="NVDA", status="WAIT", badge="⏳", stance="HARVEST", pnl=13.8):
    tech = Technicals(symbol=symbol, market="US", price=224.0, ma20=216.0,
                      ma50=200.0, rsi=54.0, hi_52w=235.0)
    z = Zone("trim", "Z2 전고", 33, 232.0, 238.0, "52주고")
    return Playbook(symbol=symbol, name=symbol, stance=stance, tech=tech,
                    zones=(z, z, z), status=status, badge=badge,
                    pnl_pct=pnl, active_zone=None)


@pytest.mark.unit
def test_dashboard_is_html() -> None:
    out = render_html.render_dashboard([_pb()])
    assert out.lstrip().startswith("<!DOCTYPE html>")
    assert "</html>" in out


@pytest.mark.unit
def test_dashboard_contains_symbol_and_rsi() -> None:
    out = render_html.render_dashboard([_pb("NVDA")])
    assert "NVDA" in out
    assert "RSI" in out


@pytest.mark.unit
def test_dashboard_sorts_triggers_first() -> None:
    pbs = [_pb("WAITER", "WAIT", "⏳"), _pb("ACTOR", "BUY_NOW", "🟢")]
    out = render_html.render_dashboard(pbs)
    assert out.index("ACTOR") < out.index("WAITER")


@pytest.mark.unit
def test_dashboard_empty_is_still_valid() -> None:
    out = render_html.render_dashboard([])
    assert "<!DOCTYPE html>" in out
