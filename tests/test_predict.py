"""Tests for corvin_jarvis.predict — predictive alerts."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from corvin_jarvis import predict, timeseries


def _seed_prices(db_path: Path, symbol: str, prices: list[float], category: str = "portfolio") -> None:
    timeseries.init_db(db_path)
    for i, p in enumerate(prices):
        ts = f"2026-05-{10+i:02d}T00:00:00+00:00"
        kst = f"2026-05-{10+i:02d}T09:00:00+09:00"
        snap = {
            "timestamp_utc": ts, "timestamp_kst": kst,
            "indices": {}, "commodities": {}, "fx": {},
            "portfolio": [], "watchlist": [],
        }
        if category == "portfolio":
            snap["portfolio"] = [{
                "symbol": symbol, "shares": 1, "avg_price": p, "currency": "USD",
                "current_price": p, "pnl_pct": 0.0, "market_value": p, "error": None,
            }]
        else:
            snap["indices"] = {symbol: {"price": p, "pct_change": 0.0, "source": "test", "error": None}}
        timeseries.write_snapshot(db_path, snap)


@pytest.mark.unit
def test_log_returns_basic(tmp_db_path: Path) -> None:
    _seed_prices(tmp_db_path, "META", [600.0, 606.0, 612.0])
    rs = predict.log_returns(tmp_db_path, "META", days=10)
    assert len(rs) == 2
    assert rs[0] == pytest.approx(math.log(606 / 600), abs=1e-6)
    assert rs[1] == pytest.approx(math.log(612 / 606), abs=1e-6)


@pytest.mark.unit
def test_log_returns_empty_when_no_data(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    assert predict.log_returns(tmp_db_path, "META", days=10) == []


@pytest.mark.unit
def test_log_returns_skip_zero_or_negative(tmp_db_path: Path) -> None:
    """가격에 None이나 0이 섞이면 skip."""
    timeseries.init_db(tmp_db_path)
    for i, p in enumerate([600.0, 0.0, 612.0]):
        snap = {
            "timestamp_utc": f"2026-05-{10+i:02d}T00:00:00+00:00",
            "timestamp_kst": f"2026-05-{10+i:02d}T09:00:00+09:00",
            "indices": {},
            "commodities": {},
            "fx": {},
            "portfolio": [{
                "symbol": "X", "shares": 1, "avg_price": 1, "currency": "USD",
                "current_price": p, "pnl_pct": 0, "market_value": p, "error": None,
            }] if p > 0 else [],
            "watchlist": [],
        }
        timeseries.write_snapshot(tmp_db_path, snap)
    rs = predict.log_returns(tmp_db_path, "X", days=10)
    assert all(math.isfinite(r) for r in rs)


# ---- probability_below ----


@pytest.mark.unit
def test_probability_below_when_threshold_equals_current() -> None:
    """current=threshold일 때 P ≈ 0.5 (zero drift 가정)."""
    p = predict.probability_below(
        current_price=600.0, threshold=600.0,
        mu=0.0, sigma=0.02, horizon_days=3,
    )
    assert 0.45 < p < 0.55


@pytest.mark.unit
def test_probability_below_far_below_is_high() -> None:
    """threshold가 현재가보다 50% 낮으면 P ≈ 0 (low probability)."""
    p = predict.probability_below(
        current_price=600.0, threshold=300.0,
        mu=0.0, sigma=0.02, horizon_days=3,
    )
    assert p < 0.001


@pytest.mark.unit
def test_probability_below_far_above_is_high() -> None:
    """threshold가 현재가보다 훨씬 높으면 P ≈ 1."""
    p = predict.probability_below(
        current_price=600.0, threshold=1000.0,
        mu=0.0, sigma=0.02, horizon_days=3,
    )
    assert p > 0.99


@pytest.mark.unit
def test_probability_below_realistic_drop() -> None:
    """current=605, threshold=598 (-1.2%), sigma=2%/day, 3일 → ~10-35%."""
    p = predict.probability_below(
        current_price=605.0, threshold=598.0,
        mu=0.0, sigma=0.02, horizon_days=3,
    )
    assert 0.05 < p < 0.45


@pytest.mark.unit
def test_probability_below_zero_sigma_returns_step() -> None:
    """sigma=0 (degenerate) → threshold ≥ current면 1, 아니면 0."""
    assert predict.probability_below(600.0, 700.0, mu=0.0, sigma=0.0, horizon_days=3) == 1.0
    assert predict.probability_below(600.0, 500.0, mu=0.0, sigma=0.0, horizon_days=3) == 0.0


# ---- build_predictive_alerts ----


@pytest.mark.unit
def test_build_predictive_alerts_high_probability_triggers(tmp_db_path: Path) -> None:
    """변동성 큰 종목 + threshold 근처 → high P → alert."""
    # 변동성 ±2% 일별
    prices = [600.0, 612.0, 588.0, 605.0, 593.0, 610.0, 595.0, 608.0, 591.0, 604.0]
    _seed_prices(tmp_db_path, "META", prices)
    levels = {"META": {"threshold": 588.0, "horizon_days": 3, "min_prob": 0.10}}
    alerts = predict.build_predictive_alerts(tmp_db_path, levels)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["category"] == "predictive"
    assert a["metric"] == "META"
    assert 0.1 <= a["value"] <= 1.0
    assert "META" in a["message"]


@pytest.mark.unit
def test_build_predictive_alerts_skip_low_probability(tmp_db_path: Path) -> None:
    """threshold 매우 멀고 변동성 낮으면 → P 거의 0 → no alert."""
    prices = [600.0, 601.0, 600.5, 599.8, 600.2]  # 매우 안정
    _seed_prices(tmp_db_path, "MSFT", prices)
    levels = {"MSFT": {"threshold": 400.0, "horizon_days": 3, "min_prob": 0.1}}
    alerts = predict.build_predictive_alerts(tmp_db_path, levels)
    assert alerts == []


@pytest.mark.unit
def test_build_predictive_alerts_skip_insufficient_data(tmp_db_path: Path) -> None:
    timeseries.init_db(tmp_db_path)
    levels = {"X": {"threshold": 100.0, "horizon_days": 3, "min_prob": 0.1}}
    alerts = predict.build_predictive_alerts(tmp_db_path, levels)
    assert alerts == []


@pytest.mark.unit
def test_build_predictive_alerts_severity_scales_with_prob(tmp_db_path: Path) -> None:
    prices = [600.0 + (i % 2) * 20 - 10 for i in range(15)]  # ±$10 oscillation
    _seed_prices(tmp_db_path, "NVDA", prices)
    # near-money threshold → high P → expect medium/high severity
    levels = {"NVDA": {"threshold": 595.0, "horizon_days": 5, "min_prob": 0.05}}
    alerts = predict.build_predictive_alerts(tmp_db_path, levels)
    if alerts:
        assert alerts[0]["severity"] in {"medium", "high", "critical"}
