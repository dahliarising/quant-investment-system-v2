"""Tests for corvin_jarvis.playbook.models."""
from __future__ import annotations

import dataclasses

import pytest

from corvin_jarvis.playbook import models


@pytest.mark.unit
def test_zone_is_frozen() -> None:
    z = models.Zone(kind="buy", label="Z1", ratio=30, low=100.0, high=103.0, note="MA20")
    with pytest.raises(dataclasses.FrozenInstanceError):
        z.ratio = 40  # type: ignore[misc]


@pytest.mark.unit
def test_playbook_holds_zones_tuple() -> None:
    tech = models.Technicals(symbol="X", market="US", price=10.0, ma20=11.0,
                             ma50=12.0, rsi=50.0, hi_52w=15.0)
    z = models.Zone(kind="buy", label="Z1", ratio=30, low=9.9, high=10.1, note="")
    pb = models.Playbook(symbol="X", name="X Co", stance="ENTER", tech=tech,
                         zones=(z,), status="BUY_NOW", badge="🟢",
                         pnl_pct=None, active_zone=z)
    assert pb.zones[0].ratio == 30
    assert pb.active_zone is z
