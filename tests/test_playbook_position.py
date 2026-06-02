"""Tests for corvin_jarvis.playbook.position."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import ladder, position
from corvin_jarvis.playbook.models import Technicals


def _tech(price=100.0, ma20=105.0, ma50=110.0, rsi=50.0, hi=130.0) -> Technicals:
    return Technicals(symbol="X", market="US", price=price, ma20=ma20,
                      ma50=ma50, rsi=rsi, hi_52w=hi)


@pytest.mark.unit
def test_stance_enter_when_not_held() -> None:
    assert position.stance(pnl_pct=None) == "ENTER"


@pytest.mark.unit
def test_stance_accumulate_when_underwater() -> None:
    assert position.stance(pnl_pct=-7.3) == "ACCUMULATE"


@pytest.mark.unit
def test_stance_harvest_when_in_profit() -> None:
    assert position.stance(pnl_pct=20.2) == "HARVEST"


@pytest.mark.unit
def test_buy_status_buy_now_when_price_in_or_below_deep() -> None:
    tech = _tech(price=100.0, ma20=105.0, ma50=110.0)  # below all buy zones
    zones = ladder.build_buy_ladder(tech)
    status, badge, active = position.classify(tech, zones, "ENTER")
    assert status == "BUY_NOW"
    assert badge == "🟢"
    assert active is not None


@pytest.mark.unit
def test_buy_status_wait_when_price_above_zones() -> None:
    tech = _tech(price=200.0, ma20=105.0, ma50=110.0)
    zones = ladder.build_buy_ladder(tech)
    status, badge, active = position.classify(tech, zones, "ENTER")
    assert status == "WAIT"
    assert active is None


@pytest.mark.unit
def test_trim_status_trim_now_when_overheated() -> None:
    tech = _tech(price=145.0, ma20=120.0, ma50=110.0, rsi=78.0, hi=145.0)
    zones = ladder.build_trim_ladder(tech)
    status, badge, active = position.classify(tech, zones, "HARVEST")
    assert status == "TRIM_NOW"
    assert badge == "✂️"


@pytest.mark.unit
def test_trim_status_invalid_when_below_ma50() -> None:
    tech = _tech(price=100.0, ma20=120.0, ma50=110.0, rsi=40.0, hi=145.0)
    zones = ladder.build_trim_ladder(tech)
    status, badge, active = position.classify(tech, zones, "HARVEST")
    assert status == "INVALID"
    assert badge == "⚠️"
