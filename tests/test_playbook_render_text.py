"""Tests for corvin_jarvis.playbook.render_text."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import render_text
from corvin_jarvis.playbook.models import Playbook, Technicals, Zone


def _pb(symbol, status, badge, stance, pnl=None) -> Playbook:
    tech = Technicals(symbol=symbol, market="US", price=100.0, ma20=105.0,
                      ma50=110.0, rsi=50.0, hi_52w=130.0)
    z = Zone("buy", "Z2 핵심지지", 40, 99.0, 101.0, "MA50")
    return Playbook(symbol=symbol, name=symbol, stance=stance, tech=tech,
                    zones=(z, z, z), status=status, badge=badge,
                    pnl_pct=pnl, active_zone=z)


@pytest.mark.unit
def test_push_lists_triggers_first() -> None:
    pbs = [
        _pb("AAA", "WAIT", "⏳", "ENTER"),
        _pb("BWXT", "BUY_NOW", "🟢", "ENTER"),
    ]
    out = render_text.render_push(pbs, date_label="6/2")
    assert "오늘 액션" in out
    assert "BWXT" in out
    # trigger appears before the wait-only summary section
    assert out.index("BWXT") < out.index("관찰")


@pytest.mark.unit
def test_push_counts_watch_and_triggers() -> None:
    pbs = [_pb("AAA", "WAIT", "⏳", "ENTER") for _ in range(5)]
    pbs.append(_pb("BBB", "BUY_NOW", "🟢", "ENTER"))
    out = render_text.render_push(pbs, date_label="6/2")
    assert "트리거1" in out or "트리거 1" in out


@pytest.mark.unit
def test_push_shows_holdings_line() -> None:
    pbs = [_pb("MSFT", "TRIM_NOW", "✂️", "HARVEST", pnl=20.2)]
    out = render_text.render_push(pbs, date_label="6/2")
    assert "보유" in out
    assert "MSFT" in out
