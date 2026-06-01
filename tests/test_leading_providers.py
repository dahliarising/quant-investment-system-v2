"""Tests for corvin_jarvis.leading_providers (mock fetcher, no real calls)."""
from __future__ import annotations

import pytest

from corvin_jarvis import leading_providers as lp
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_ensemble_provider_builds_signals():
    closes = {"012450": [float(i) for i in range(1, 261)]}
    bench = [100.0] * 261

    def price_fetcher(sym, days=252):
        return closes.get(sym, [])

    def bench_fetcher():
        return bench

    sigs = lp.ensemble_provider(
        ["012450"], price_fetcher=price_fetcher, bench_fetcher=bench_fetcher
    )
    assert len(sigs) == 1
    assert isinstance(sigs[0], LeadingSignal)
    assert sigs[0].pillar == "ensemble"


@pytest.mark.unit
def test_ensemble_provider_uses_volume_fetcher():
    closes = {"012450": [float(i) for i in range(1, 261)]}

    def price_fetcher(sym, days=252):
        return closes.get(sym, [])

    def bench_fetcher():
        return [100.0] * 261

    def volume_fetcher(sym):
        return (300.0, 100.0)   # 3배 돌파

    sigs = lp.ensemble_provider(
        ["012450"], price_fetcher=price_fetcher,
        bench_fetcher=bench_fetcher, volume_fetcher=volume_fetcher,
    )
    assert len(sigs) == 1
    assert sigs[0].evidence["volume"] >= 80


@pytest.mark.unit
def test_ensemble_provider_skips_insufficient_history():
    def price_fetcher(sym, days=252):
        return [100.0, 101.0]

    def bench_fetcher():
        return [100.0, 101.0]

    sigs = lp.ensemble_provider(
        ["012450"], price_fetcher=price_fetcher, bench_fetcher=bench_fetcher
    )
    assert sigs == []


@pytest.mark.unit
def test_cross_market_provider():
    targets = [("012450", "LMT 야간", "LMT")]
    quotes = {"LMT": 3.0}

    def pct_fetcher(proxy_symbol):
        return quotes.get(proxy_symbol)

    sigs = lp.cross_market_provider(targets, pct_fetcher=pct_fetcher)
    assert len(sigs) == 1
    assert sigs[0].pillar == "cross_market"
    assert sigs[0].advisory is True


@pytest.mark.unit
def test_cross_market_provider_skips_missing_proxy():
    targets = [("012450", "LMT 야간", "LMT")]

    def pct_fetcher(proxy_symbol):
        return None

    assert lp.cross_market_provider(targets, pct_fetcher=pct_fetcher) == []


@pytest.mark.unit
def test_fundamental_provider_builds():
    metrics = {"012450": {"eps_growth_yoy": 0.25, "revenue_growth_yoy": 0.15}}

    def metrics_fetcher(sym):
        return metrics.get(sym)

    def tone_fetcher(sym):
        return 0.3

    sigs = lp.fundamental_provider(
        ["012450"], metrics_fetcher=metrics_fetcher, tone_fetcher=tone_fetcher,
    )
    assert len(sigs) == 1
    assert sigs[0].pillar == "fundamental"
    assert sigs[0].evidence["qualitative_used"] is False


@pytest.mark.unit
def test_fundamental_provider_skips_no_metrics():
    def metrics_fetcher(sym):
        return None

    def tone_fetcher(sym):
        return 0.0

    sigs = lp.fundamental_provider(
        ["012450"], metrics_fetcher=metrics_fetcher, tone_fetcher=tone_fetcher,
    )
    assert sigs == []
