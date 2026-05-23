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


# ---- compute_correlation ----


def _seed_pair_history(db_path: Path, sym_a: str, prices_a: list[float],
                       sym_b: str, prices_b: list[float]) -> None:
    from corvin_jarvis import timeseries
    timeseries.init_db(db_path)
    assert len(prices_a) == len(prices_b)
    for i, (a, b) in enumerate(zip(prices_a, prices_b)):
        snap = {
            "timestamp_utc": f"2026-05-{10+i:02d}T00:00:00+00:00",
            "timestamp_kst": f"2026-05-{10+i:02d}T09:00:00+09:00",
            "indices": {
                sym_a: {"price": a, "pct_change": 0.0, "source": "test", "error": None},
                sym_b: {"price": b, "pct_change": 0.0, "source": "test", "error": None},
            },
            "commodities": {}, "fx": {}, "portfolio": [], "watchlist": [],
        }
        timeseries.write_snapshot(db_path, snap)


@pytest.mark.unit
def test_correlation_positive(tmp_db_path: Path) -> None:
    _seed_pair_history(
        tmp_db_path,
        "kospi", [3000.0, 3050.0, 3100.0, 3120.0, 3150.0],
        "sp500", [5000.0, 5060.0, 5120.0, 5140.0, 5180.0],
    )
    c = regime.compute_correlation(tmp_db_path, "kospi", "sp500", days=10)
    assert c is not None
    assert c > 0.9  # strongly positive


@pytest.mark.unit
def test_correlation_negative(tmp_db_path: Path) -> None:
    _seed_pair_history(
        tmp_db_path,
        "kospi", [3000.0, 3050.0, 3100.0, 3120.0, 3150.0],
        "sp500", [5200.0, 5150.0, 5100.0, 5060.0, 5000.0],
    )
    c = regime.compute_correlation(tmp_db_path, "kospi", "sp500", days=10)
    assert c is not None
    assert c < -0.9


@pytest.mark.unit
def test_correlation_returns_none_with_insufficient_data(tmp_db_path: Path) -> None:
    from corvin_jarvis import timeseries
    timeseries.init_db(tmp_db_path)
    c = regime.compute_correlation(tmp_db_path, "X", "Y", days=10)
    assert c is None


@pytest.mark.unit
def test_correlation_handles_constant_series(tmp_db_path: Path) -> None:
    _seed_pair_history(
        tmp_db_path,
        "a", [100.0, 100.0, 100.0, 100.0, 100.0],
        "b", [200.0, 210.0, 205.0, 215.0, 220.0],
    )
    c = regime.compute_correlation(tmp_db_path, "a", "b", days=10)
    assert c is None  # zero variance — undefined
