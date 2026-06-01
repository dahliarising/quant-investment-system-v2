"""Tests for corvin_jarvis.market_data (pure parse, no real API)."""
from __future__ import annotations

import pytest

from corvin_jarvis import market_data as md


@pytest.mark.unit
def test_is_kr():
    assert md._is_kr("012450") is True
    assert md._is_kr("META") is False
