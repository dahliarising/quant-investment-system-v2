"""Tests for corvin_jarvis.signals.leading_signal."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals.leading_signal import (
    LeadingSignal,
    apply_confidence_gate,
)


def _sig(conf, advisory=False, direction="bull"):
    return LeadingSignal(
        pillar="event", symbol="012450", direction=direction,
        confidence=conf, score=None, horizon="days",
        advisory=advisory, message="x", evidence={},
    )


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


@pytest.mark.unit
def test_gate_mutes_below_threshold():
    sigs = [_sig(40.0), _sig(59.9), _sig(60.0), _sig(85.0)]
    passed = apply_confidence_gate(sigs, threshold=60.0)
    assert [s.confidence for s in passed] == [60.0, 85.0]


@pytest.mark.unit
def test_gate_advisory_never_solo():
    # advisory 신호는 게이트를 통과해도 solo=False (브리프 내 한 줄로만)
    sigs = [_sig(90.0, advisory=True)]
    passed = apply_confidence_gate(sigs, threshold=60.0)
    assert len(passed) == 1
    assert passed[0].advisory is True


@pytest.mark.unit
def test_gate_empty_returns_empty():
    assert apply_confidence_gate([], threshold=60.0) == []
