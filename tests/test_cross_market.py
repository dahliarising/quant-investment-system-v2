"""Tests for corvin_jarvis.signals.cross_market."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals import cross_market
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_cross_market_signal_is_advisory():
    sig = cross_market.build_cross_market_signal(
        symbol="012450", proxy_name="LMT/RTX/ITA 야간", proxy_pct=3.0,
    )
    assert isinstance(sig, LeadingSignal)
    assert sig.pillar == "cross_market"
    assert sig.advisory is True
    assert sig.direction == "bull"


@pytest.mark.unit
def test_cross_market_confidence_penalized():
    sig = cross_market.build_cross_market_signal(
        symbol="012450", proxy_name="LMT 야간", proxy_pct=5.0,
    )
    # base = clamp(|5|*10)=50, penalized = 50*0.7 = 35
    assert abs(sig.confidence - 35.0) < 0.01


@pytest.mark.unit
def test_cross_market_bear_on_negative_proxy():
    sig = cross_market.build_cross_market_signal(
        symbol="META", proxy_name="NQ 선물", proxy_pct=-4.0,
    )
    assert sig.direction == "bear"


@pytest.mark.unit
def test_cross_market_neutral_small_move():
    sig = cross_market.build_cross_market_signal(
        symbol="012450", proxy_name="ITA 야간", proxy_pct=0.3,
    )
    assert sig.direction == "neutral"
