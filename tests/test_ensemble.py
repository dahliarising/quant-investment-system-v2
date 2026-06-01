"""Tests for corvin_jarvis.signals.ensemble."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals import ensemble
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_minervini_all_criteria_pass():
    score = ensemble.minervini_trend_score(
        price=120.0, ma50=110.0, ma150=100.0, ma200=90.0,
        low_52w=70.0, high_52w=125.0,
    )
    assert score == 100.0


@pytest.mark.unit
def test_minervini_downtrend_low_score():
    score = ensemble.minervini_trend_score(
        price=80.0, ma50=90.0, ma150=100.0, ma200=110.0,
        low_52w=78.0, high_52w=160.0,
    )
    assert score <= 35.0


@pytest.mark.unit
def test_minervini_handles_none():
    score = ensemble.minervini_trend_score(
        price=100.0, ma50=None, ma150=None, ma200=None,
        low_52w=None, high_52w=None,
    )
    assert score == 50.0


@pytest.mark.unit
def test_volume_breakthrough_high():
    assert ensemble.volume_breakthrough_score(vol=200.0, vol_avg=100.0) >= 80


@pytest.mark.unit
def test_volume_breakthrough_normal():
    assert 40 <= ensemble.volume_breakthrough_score(vol=100.0, vol_avg=100.0) <= 60


@pytest.mark.unit
def test_volume_breakthrough_none():
    assert ensemble.volume_breakthrough_score(vol=None, vol_avg=100.0) == 50.0


@pytest.mark.unit
def test_multi_timeframe_rs_strong():
    rs = {"1w": 3.0, "1m": 5.0, "3m": 8.0, "6m": 12.0}
    assert ensemble.multi_timeframe_rs_score(rs) >= 70


@pytest.mark.unit
def test_multi_timeframe_rs_weak():
    rs = {"1w": -3.0, "1m": -5.0, "3m": -8.0, "6m": -12.0}
    assert ensemble.multi_timeframe_rs_score(rs) <= 30


@pytest.mark.unit
def test_multi_timeframe_rs_empty():
    assert ensemble.multi_timeframe_rs_score({}) == 50.0


@pytest.mark.unit
def test_ensemble_score_weighted():
    score = ensemble.ensemble_score(minervini=100.0, rs=75.0, volume=50.0)
    assert abs(score - 80.0) < 0.01


@pytest.mark.unit
def test_build_ensemble_signal_bull():
    sig = ensemble.build_ensemble_signal(
        symbol="012450", minervini=90.0, rs=80.0, volume=70.0,
    )
    assert isinstance(sig, LeadingSignal)
    assert sig.pillar == "ensemble"
    assert sig.direction == "bull"
    assert sig.horizon == "days"
    assert sig.advisory is False
    assert sig.score is not None and sig.score >= 70
    assert sig.confidence >= 60


@pytest.mark.unit
def test_build_ensemble_signal_bear():
    sig = ensemble.build_ensemble_signal(
        symbol="META", minervini=20.0, rs=25.0, volume=40.0,
    )
    assert sig.direction == "bear"


@pytest.mark.unit
def test_indicators_from_closes():
    closes = [float(i) for i in range(1, 261)]  # 1..260 상승추세
    ind = ensemble.indicators_from_closes(closes)
    assert ind["price"] == 260.0
    assert ind["ma50"] is not None and ind["ma50"] < 260.0
    assert ind["high_52w"] == 260.0
    assert ind["low_52w"] == closes[-252]


@pytest.mark.unit
def test_indicators_from_closes_short_series():
    ind = ensemble.indicators_from_closes([100.0, 101.0])
    assert ind["price"] == 101.0
    assert ind["ma50"] is None
    assert ind["ma200"] is None


@pytest.mark.unit
def test_rs_windows_from_closes():
    sym = [100.0] * 200 + [110.0]
    bench = [100.0] * 201
    rs = ensemble.rs_windows_from_closes(sym, bench)
    assert rs["1w"] > 0
    assert all(k in rs for k in ("1w", "1m", "3m", "6m"))


@pytest.mark.unit
def test_rs_windows_insufficient():
    assert ensemble.rs_windows_from_closes([100.0], [100.0]) == {}


@pytest.mark.unit
def test_volume_stats_basic():
    vols = [100.0] * 19 + [200.0]   # avg20=105, latest=200
    latest, avg20 = ensemble.volume_stats(vols)
    assert latest == 200.0
    assert abs(avg20 - 105.0) < 1e-9


@pytest.mark.unit
def test_volume_stats_short():
    latest, avg20 = ensemble.volume_stats([])
    assert latest is None and avg20 is None
