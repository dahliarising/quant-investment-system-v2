"""Tests for corvin_jarvis.playbook.technicals."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import technicals as t


@pytest.mark.unit
def test_sma_simple() -> None:
    assert t.sma([10, 20, 30], 3) == 20.0


@pytest.mark.unit
def test_sma_uses_last_n() -> None:
    assert t.sma([1, 2, 100, 200], 2) == 150.0


@pytest.mark.unit
def test_sma_insufficient_returns_none() -> None:
    assert t.sma([1, 2], 3) is None


@pytest.mark.unit
def test_rsi_all_up_is_100() -> None:
    prices = list(range(1, 30))  # strictly increasing
    assert t.rsi(prices) == 100.0


@pytest.mark.unit
def test_rsi_midrange_for_mixed() -> None:
    prices = [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10]
    r = t.rsi(prices)
    assert 30.0 < r < 70.0


@pytest.mark.unit
def test_recent_high() -> None:
    assert t.recent_high([5, 9, 3, 7], 252) == 9.0
