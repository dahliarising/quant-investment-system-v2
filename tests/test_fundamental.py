"""Tests for corvin_jarvis.signals.fundamental."""
from __future__ import annotations

import pytest

from corvin_jarvis.signals import fundamental
from corvin_jarvis.signals.leading_signal import LeadingSignal


@pytest.mark.unit
def test_financial_growth_score_strong():
    metrics = {
        "eps_growth_yoy": 0.30,
        "op_margin_trend": 0.05,
        "debt_ratio": 0.3,
        "revenue_growth_yoy": 0.20,
    }
    score = fundamental.financial_growth_score(metrics)
    assert 70 <= score <= 100


@pytest.mark.unit
def test_financial_growth_score_weak():
    metrics = {
        "eps_growth_yoy": -0.20,
        "op_margin_trend": -0.05,
        "debt_ratio": 2.5,
        "revenue_growth_yoy": -0.10,
    }
    score = fundamental.financial_growth_score(metrics)
    assert 0 <= score <= 35


@pytest.mark.unit
def test_financial_growth_score_partial_metrics():
    score = fundamental.financial_growth_score({"eps_growth_yoy": 0.15})
    assert 0 <= score <= 100


@pytest.mark.unit
def test_financial_growth_score_empty_returns_neutral():
    assert fundamental.financial_growth_score({}) == 50.0


@pytest.mark.unit
def test_news_sentiment_score_positive():
    assert fundamental.news_sentiment_score(0.5) > 60


@pytest.mark.unit
def test_news_sentiment_score_negative():
    assert fundamental.news_sentiment_score(-0.5) < 40


@pytest.mark.unit
def test_news_sentiment_score_none_returns_neutral():
    assert fundamental.news_sentiment_score(None) == 50.0


@pytest.mark.unit
def test_news_sentiment_score_clamps():
    assert fundamental.news_sentiment_score(5.0) == 100.0
    assert fundamental.news_sentiment_score(-5.0) == 0.0


@pytest.mark.unit
def test_combine_growth_all_present():
    score = fundamental.combine_growth_score(
        financial=80.0, qualitative=70.0, news=60.0
    )
    assert abs(score - 73.0) < 0.01


@pytest.mark.unit
def test_combine_growth_qualitative_none_renormalizes():
    score = fundamental.combine_growth_score(
        financial=80.0, qualitative=None, news=60.0
    )
    assert abs(score - 74.29) < 0.1


@pytest.mark.unit
def test_growth_to_direction():
    assert fundamental.growth_to_direction(75.0) == "bull"
    assert fundamental.growth_to_direction(50.0) == "neutral"
    assert fundamental.growth_to_direction(30.0) == "bear"


@pytest.mark.unit
def test_build_fundamental_signal_bull():
    sig = fundamental.build_fundamental_signal(
        symbol="012450", financial=85.0, qualitative=80.0, news=70.0,
    )
    assert isinstance(sig, LeadingSignal)
    assert sig.pillar == "fundamental"
    assert sig.symbol == "012450"
    assert sig.direction == "bull"
    assert sig.score is not None and sig.score >= 70
    assert sig.horizon == "weeks"
    assert sig.advisory is False
    assert sig.confidence >= 60


@pytest.mark.unit
def test_build_fundamental_signal_neutral_low_confidence():
    sig = fundamental.build_fundamental_signal(
        symbol="META", financial=52.0, qualitative=50.0, news=48.0,
    )
    assert sig.direction == "neutral"
    assert sig.confidence < 60


@pytest.mark.unit
def test_build_fundamental_signal_degrades_without_qualitative():
    sig = fundamental.build_fundamental_signal(
        symbol="NVDA", financial=80.0, qualitative=None, news=70.0,
    )
    assert sig.evidence["qualitative_used"] is False
