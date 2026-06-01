"""Tests for corvin_jarvis.signals.leading_signal."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_leading_signal_is_frozen():
    sig = LeadingSignal(
        pillar="event", symbol="012450", direction="neutral",
        confidence=72.0, score=None, horizon="days",
        advisory=False, message="실적 D-3", evidence={"days_to": 3},
    )
    with pytest.raises(Exception):
        sig.confidence = 10.0  # frozen → 변경 불가


@pytest.mark.unit
def test_leading_signal_rejects_bad_direction():
    with pytest.raises(ValueError):
        LeadingSignal(
            pillar="event", symbol="012450", direction="up",  # invalid
            confidence=50.0, score=None, horizon="days",
            advisory=False, message="x", evidence={},
        )


@pytest.mark.unit
def test_leading_signal_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        LeadingSignal(
            pillar="event", symbol="012450", direction="bull",
            confidence=150.0, score=None, horizon="days",  # >100
            advisory=False, message="x", evidence={},
        )
