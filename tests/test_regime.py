"""Tests for corvin_jarvis.regime detector."""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis import regime


@pytest.mark.unit
def test_label_low_vix_stable_fx_is_risk_on() -> None:
    out = regime.label_from_signals(vix=12.0, vix_zscore=-1.0, usd_krw_pct=0.1, correlation=None)
    assert out["label"] == "risk_on"
    assert -100 <= out["score"] <= 100
    assert out["score"] > 30


@pytest.mark.unit
def test_label_high_vix_is_risk_off() -> None:
    out = regime.label_from_signals(vix=30.0, vix_zscore=2.0, usd_krw_pct=1.5, correlation=None)
    assert out["label"] == "risk_off"
    assert out["score"] < -30


@pytest.mark.unit
def test_label_extreme_vix_is_crisis() -> None:
    out = regime.label_from_signals(vix=45.0, vix_zscore=3.5, usd_krw_pct=3.0, correlation=None)
    assert out["label"] == "crisis"


@pytest.mark.unit
def test_label_moderate_is_neutral() -> None:
    out = regime.label_from_signals(vix=18.0, vix_zscore=0.0, usd_krw_pct=0.3, correlation=None)
    assert out["label"] == "neutral"


@pytest.mark.unit
def test_label_includes_drivers() -> None:
    out = regime.label_from_signals(vix=32.0, vix_zscore=2.5, usd_krw_pct=2.1, correlation=None)
    drivers = out["drivers"]
    assert any("VIX" in d for d in drivers)
    assert any("USD/KRW" in d or "원" in d for d in drivers)
