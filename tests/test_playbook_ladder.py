"""Tests for corvin_jarvis.playbook.ladder."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import ladder
from corvin_jarvis.playbook.models import Technicals


def _tech(price=100.0, ma20=105.0, ma50=110.0, rsi=50.0, hi=130.0) -> Technicals:
    return Technicals(symbol="X", market="US", price=price, ma20=ma20,
                      ma50=ma50, rsi=rsi, hi_52w=hi)


@pytest.mark.unit
def test_buy_ladder_has_three_zones_30_40_30() -> None:
    zones = ladder.build_buy_ladder(_tech())
    assert [z.ratio for z in zones] == [30, 40, 30]
    assert all(z.kind == "buy" for z in zones)


@pytest.mark.unit
def test_buy_ladder_levels_anchor_to_ma() -> None:
    zones = ladder.build_buy_ladder(_tech(ma20=105.0, ma50=110.0))
    z1, z2, z3 = zones
    assert z1.low < 105.0 < z1.high      # Z1 around MA20
    assert z2.low < 110.0 < z2.high      # Z2 around MA50
    assert z3.high < 110.0               # Z3 below MA50 (deep)


@pytest.mark.unit
def test_trim_ladder_has_three_zones_33_33_34() -> None:
    zones = ladder.build_trim_ladder(_tech())
    assert [z.ratio for z in zones] == [33, 33, 34]
    assert all(z.kind == "trim" for z in zones)


@pytest.mark.unit
def test_trim_z2_anchors_to_52w_high() -> None:
    zones = ladder.build_trim_ladder(_tech(hi=130.0))
    z2 = zones[1]
    assert z2.low <= 130.0 <= z2.high
