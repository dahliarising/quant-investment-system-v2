"""Tests for corvin_jarvis.dca_timing — daily DCA buy candidate scoring."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from corvin_jarvis import dca_timing


# ---------- 가격 시계열 helper ----------


def _flat(value: float, n: int = 60) -> list[float]:
    return [value] * n


def _linear(start: float, end: float, n: int) -> list[float]:
    if n < 2:
        return [start]
    step = (end - start) / (n - 1)
    return [start + step * i for i in range(n)]


# ---------- 단위 신호 ----------


@pytest.mark.unit
def test_rsi_uptrend_above_70() -> None:
    """단조 상승 시 RSI는 100에 근접."""
    prices = _linear(100.0, 200.0, 30)
    rsi = dca_timing._rsi(prices, period=14)
    assert rsi is not None and rsi >= 99.0


@pytest.mark.unit
def test_rsi_downtrend_below_30() -> None:
    """단조 하락 시 RSI는 0에 근접."""
    prices = _linear(200.0, 100.0, 30)
    rsi = dca_timing._rsi(prices, period=14)
    assert rsi is not None and rsi <= 1.0


@pytest.mark.unit
def test_rsi_insufficient_history() -> None:
    assert dca_timing._rsi([1.0, 2.0, 3.0], period=14) is None


@pytest.mark.unit
def test_ma_distance_negative_when_price_below_ma() -> None:
    prices = [100.0] * 49 + [90.0]
    dist = dca_timing._ma_distance_pct(prices, window=50)
    assert dist is not None
    # MA50 = (100*49 + 90)/50 = 99.8, distance = (90-99.8)/99.8 = -9.82%
    assert dist == pytest.approx(-9.82, abs=0.05)


@pytest.mark.unit
def test_ma_distance_positive_when_price_above_ma() -> None:
    prices = [100.0] * 49 + [110.0]
    dist = dca_timing._ma_distance_pct(prices, window=50)
    assert dist is not None and dist > 0


@pytest.mark.unit
def test_zscore_negative_for_dip() -> None:
    """일정 가격 후 마지막에 큰 폭 하락 → 음의 z-score."""
    prices = [100.0] * 19 + [95.0]
    z = dca_timing._zscore(prices, window=20)
    assert z is not None and z < -1.0


@pytest.mark.unit
def test_drawdown_from_52w_high() -> None:
    prices = [50.0] + [200.0] + [150.0] * 250  # 200 peak, current 150 → -25%
    dd = dca_timing._drawdown_52w_pct(prices)
    assert dd == pytest.approx(-25.0, abs=0.1)


# ---------- composite scoring ----------


@pytest.mark.unit
def test_composite_score_oversold_combo() -> None:
    """RSI<30 + MA50<-10% + z<-2 + 52wDD<-15% → 최대 점수 누적."""
    score, bd = dca_timing._composite_score(
        rsi=25.0, ma_dist=-12.0, z=-2.5, dd_52w=-20.0,
    )
    assert score == 95  # 30 + 30 + 25 + 10 = 95
    assert bd == {"rsi": 30, "ma50": 30, "zscore": 25, "drawdown_52w": 10}


@pytest.mark.unit
def test_composite_score_neutral() -> None:
    score, _ = dca_timing._composite_score(
        rsi=55.0, ma_dist=2.0, z=0.5, dd_52w=-5.0,
    )
    assert score == 0


@pytest.mark.unit
def test_composite_score_partial_signal() -> None:
    """RSI 38만 약한 oversold → 15점만."""
    score, bd = dca_timing._composite_score(
        rsi=38.0, ma_dist=-3.0, z=-0.5, dd_52w=-10.0,
    )
    assert score == 15
    assert bd["rsi"] == 15


@pytest.mark.unit
def test_composite_score_caps_at_100() -> None:
    """모든 buckets 최대치 합산이 100을 넘어도 cap."""
    score, _ = dca_timing._composite_score(
        rsi=10.0, ma_dist=-20.0, z=-3.0, dd_52w=-30.0,
    )
    assert score <= 100


# ---------- Value-tilted multiplier (A2) ----------


@pytest.mark.unit
@pytest.mark.parametrize(
    "score,expected_mult",
    [
        (50, 1.0),
        (59, 1.0),
        (60, 1.5),
        (74, 1.5),
        (75, 2.0),
        (95, 2.0),
    ],
)
def test_score_to_multiplier_value_tilt(score: int, expected_mult: float) -> None:
    assert dca_timing._score_to_multiplier(score) == expected_mult


# ---------- Regime gate (composite × regime_mult) ----------


@pytest.mark.unit
def test_regime_multiplier_crisis_amplifies() -> None:
    assert dca_timing._regime_multiplier("CRISIS") == 1.3


@pytest.mark.unit
def test_regime_multiplier_euphoria_dampens() -> None:
    assert dca_timing._regime_multiplier("EUPHORIA") == 0.7


@pytest.mark.unit
def test_regime_multiplier_unknown_neutral() -> None:
    assert dca_timing._regime_multiplier(None) == 1.0
    assert dca_timing._regime_multiplier("NEUTRAL") == 1.0


@pytest.mark.unit
def test_regime_multiplier_case_insensitive() -> None:
    """regime.py는 lowercase ('crisis')로 반환 — 매핑 깨지지 않아야."""
    assert dca_timing._regime_multiplier("crisis") == 1.3
    assert dca_timing._regime_multiplier("risk_off") == 1.15
    assert dca_timing._regime_multiplier("euphoria") == 0.7


# ---------- Tier-weighted allocation (C2) ----------


@pytest.mark.unit
def test_base_allocation_tier1_dominant() -> None:
    """일 예산 ₩100,000, Tier 1 = 10종 → 종목당 ₩5,000 base (50% / 10)."""
    base = dca_timing._base_allocation_per_ticker(
        daily_budget_krw=100_000,
        tier=1,
        tier_counts={1: 10, 2: 10, 3: 10},
    )
    assert base == 5_000


@pytest.mark.unit
def test_base_allocation_tier3_smallest() -> None:
    base = dca_timing._base_allocation_per_ticker(
        daily_budget_krw=100_000,
        tier=3,
        tier_counts={1: 10, 2: 10, 3: 10},
    )
    # 15% / 10 = 1,500
    assert base == 1_500


@pytest.mark.unit
def test_base_allocation_handles_zero_tier_count() -> None:
    """해당 tier에 종목 0개 → 0 반환 (ZeroDivisionError 금지)."""
    base = dca_timing._base_allocation_per_ticker(
        daily_budget_krw=100_000,
        tier=2,
        tier_counts={1: 10, 2: 0, 3: 10},
    )
    assert base == 0


# ---------- Market filter (D3: KR/US 분리) ----------


@pytest.mark.unit
def test_is_kr_symbol_six_digits() -> None:
    assert dca_timing._is_kr_symbol("005930") is True
    assert dca_timing._is_kr_symbol("000660") is True
    assert dca_timing._is_kr_symbol("NVDA") is False
    assert dca_timing._is_kr_symbol("12345") is False  # 5 digits


# ---------- end-to-end report (mocked fetcher) ----------


def _fake_fetcher_factory(by_symbol: dict[str, list[float]]) -> Any:
    """symbol → 가격 리스트 매핑으로 fetch_daily_bars mock."""

    def _fetch(symbol: str, days: int = 252) -> list[float]:
        return by_symbol.get(symbol, [])

    return _fetch


@pytest.mark.unit
def test_build_dca_report_filters_below_threshold() -> None:
    """score < 50 종목은 candidates에 들어가지 않고 skipped에 사유 기록."""
    universe = [
        {"symbol": "NVDA", "tier": 1},
        {"symbol": "META", "tier": 1},
    ]
    by_symbol = {
        # 명백한 매수 신호 (단조 하락 + low price)
        "NVDA": _linear(200.0, 100.0, 60),
        # 별 신호 없음 (flat)
        "META": _flat(500.0, 60),
    }
    report = dca_timing.build_dca_report(
        universe=universe,
        daily_budget_krw=100_000,
        regime=None,
        fetcher=_fake_fetcher_factory(by_symbol),
    )
    syms_in = [c.symbol for c in report.candidates]
    syms_out = [s["symbol"] for s in report.skipped]
    assert "NVDA" in syms_in
    assert "META" in syms_out


@pytest.mark.unit
def test_build_dca_report_skips_insufficient_history() -> None:
    universe = [{"symbol": "TOO_NEW", "tier": 2}]
    by_symbol = {"TOO_NEW": [100.0, 101.0, 99.0]}  # 3 < MIN_HISTORY_DAYS
    report = dca_timing.build_dca_report(
        universe=universe,
        daily_budget_krw=100_000,
        fetcher=_fake_fetcher_factory(by_symbol),
    )
    assert report.candidates == []
    assert report.skipped[0]["reason"] == "insufficient_history"


@pytest.mark.unit
def test_build_dca_report_market_window_filters_kr_only() -> None:
    universe = [
        {"symbol": "NVDA", "tier": 1},
        {"symbol": "005930", "tier": 1},
    ]
    by_symbol = {
        "NVDA": _linear(200.0, 100.0, 60),
        "005930": _linear(80000.0, 50000.0, 60),
    }
    report = dca_timing.build_dca_report(
        universe=universe,
        daily_budget_krw=100_000,
        market_window="KR",
        fetcher=_fake_fetcher_factory(by_symbol),
    )
    syms = {c.symbol for c in report.candidates} | {s["symbol"] for s in report.skipped}
    assert syms == {"005930"}  # NVDA filtered out at window stage


@pytest.mark.unit
def test_build_dca_report_regime_amplifies_score() -> None:
    """동일 데이터에서 CRISIS regime이 더 많은 candidate 생성."""
    # weak dip — score ~ 45 normally (below 50 threshold)
    weak_dip_prices = _flat(100.0, 50) + _linear(100.0, 92.0, 10)
    universe = [{"symbol": "X", "tier": 1}]
    fetcher = _fake_fetcher_factory({"X": weak_dip_prices})

    neutral = dca_timing.build_dca_report(
        universe=universe, daily_budget_krw=100_000, regime="NEUTRAL", fetcher=fetcher,
    )
    crisis = dca_timing.build_dca_report(
        universe=universe, daily_budget_krw=100_000, regime="CRISIS", fetcher=fetcher,
    )
    # CRISIS는 ≥ NEUTRAL score
    assert (
        (len(crisis.candidates) >= len(neutral.candidates))
        and (sum(c.score for c in crisis.candidates) >= sum(c.score for c in neutral.candidates))
    )


@pytest.mark.unit
def test_format_discord_message_empty() -> None:
    """후보 없을 때 명확한 빈 메시지."""
    report = dca_timing.DCAReport(
        generated_at="2026-05-24T08:00:00+09:00",
        market_window="KR",
        regime="NEUTRAL",
        regime_multiplier=1.0,
        threshold=50,
        daily_budget_krw=100_000,
        candidates=[],
        skipped=[{"symbol": "005930", "reason": "below_threshold_score_20"}],
    )
    msg = dca_timing.format_discord_message(report)
    assert "DCA 신호" in msg or "후보 없음" in msg
    assert "KR" in msg or "🇰🇷" in msg


@pytest.mark.unit
def test_format_discord_message_with_candidates() -> None:
    cand = dca_timing.TickerScore(
        symbol="NVDA", market="US", tier=1, score=85,
        rsi=28.0, ma50_distance_pct=-12.5, zscore_20d=-2.1, drawdown_52w_pct=-22.0,
        current_price=100.0, multiplier=2.0, allocation_krw=10_000,
        breakdown={"rsi": 30, "ma50": 30, "zscore": 25, "drawdown_52w": 10},
    )
    report = dca_timing.DCAReport(
        generated_at="2026-05-24T21:00:00+09:00",
        market_window="US", regime="RISK_OFF", regime_multiplier=1.15,
        threshold=50, daily_budget_krw=100_000,
        candidates=[cand], skipped=[],
    )
    msg = dca_timing.format_discord_message(report)
    assert "NVDA" in msg
    assert "85" in msg  # score
    assert "10,000" in msg or "10000" in msg  # allocation


# ---------- live price (KIS real-time) ----------


@pytest.mark.unit
def test_build_dca_report_attaches_live_price() -> None:
    """live_fetcher 제공 시 candidate에 live_price + delta 채워짐."""
    universe = [{"symbol": "NVDA", "tier": 1}]
    by_symbol = {"NVDA": _linear(200.0, 100.0, 60)}  # 강한 dip → score ≥ 50

    def live_fetcher(symbol: str) -> float | None:
        return 105.0  # daily close 100과 +5% 차이 예상

    report = dca_timing.build_dca_report(
        universe=universe, daily_budget_krw=100_000,
        fetcher=_fake_fetcher_factory(by_symbol),
        live_fetcher=live_fetcher,
    )
    assert len(report.candidates) == 1
    c = report.candidates[0]
    assert c.live_price == 105.0
    assert c.live_delta_pct is not None and 4.5 <= c.live_delta_pct <= 5.5


@pytest.mark.unit
def test_build_dca_report_handles_live_fetcher_returning_none() -> None:
    """live API 실패 시 candidate는 정상 생성, live_price만 None."""
    universe = [{"symbol": "NVDA", "tier": 1}]
    by_symbol = {"NVDA": _linear(200.0, 100.0, 60)}
    report = dca_timing.build_dca_report(
        universe=universe, daily_budget_krw=100_000,
        fetcher=_fake_fetcher_factory(by_symbol),
        live_fetcher=lambda sym: None,
    )
    assert len(report.candidates) == 1
    assert report.candidates[0].live_price is None
    assert report.candidates[0].live_delta_pct is None


@pytest.mark.unit
def test_format_discord_message_shows_live_delta() -> None:
    cand = dca_timing.TickerScore(
        symbol="CCJ", market="US", tier=2, score=60,
        rsi=34.0, ma50_distance_pct=-7.5, zscore_20d=-1.4, drawdown_52w_pct=-22.0,
        current_price=104.0, multiplier=1.5, allocation_krw=3_280,
        breakdown={"rsi": 15, "ma50": 20, "zscore": 15, "drawdown_52w": 10},
        live_price=105.50, live_delta_pct=1.44,
    )
    report = dca_timing.DCAReport(
        generated_at="2026-05-24T21:00:00+09:00", market_window="US",
        regime="neutral", regime_multiplier=1.0, threshold=50,
        daily_budget_krw=50_000, candidates=[cand], skipped=[],
    )
    msg = dca_timing.format_discord_message(report)
    assert "실시간" in msg
    assert "105.50" in msg
    assert "+1.44" in msg or "1.44" in msg


# ---------- skip-reason transparency (below_threshold vs data error) ----------


@pytest.mark.unit
def test_skip_summary_distinguishes_data_error() -> None:
    skipped = [
        {"symbol": "A", "reason": "below_threshold_score_45"},
        {"symbol": "B", "reason": "below_threshold_score_0"},
        {"symbol": "C", "reason": "insufficient_history"},
    ]
    below, data_err = dca_timing._skip_summary(skipped)
    assert below == 2
    assert data_err == 1


@pytest.mark.unit
def test_empty_message_below_threshold_not_flagged_as_outage() -> None:
    """후보 없음 + 전부 점수미달 → '데이터오류' 문구 없어야 (오진 방지)."""
    report = dca_timing.DCAReport(
        generated_at="2026-05-26T15:40:00+09:00", market_window="KR",
        regime="NEUTRAL", regime_multiplier=1.0, threshold=50, daily_budget_krw=50_000,
        candidates=[],
        skipped=[{"symbol": "012450", "reason": "below_threshold_score_45"}],
    )
    msg = dca_timing.format_discord_message(report)
    assert "점수미달" in msg
    assert "데이터오류" not in msg


@pytest.mark.unit
def test_empty_message_flags_real_data_error() -> None:
    """fetch 실패(insufficient_history) 있으면 데이터오류로 명확히 경고."""
    report = dca_timing.DCAReport(
        generated_at="2026-05-26T15:40:00+09:00", market_window="KR",
        regime="NEUTRAL", regime_multiplier=1.0, threshold=50, daily_budget_krw=50_000,
        candidates=[],
        skipped=[{"symbol": "012450", "reason": "insufficient_history"}],
    )
    msg = dca_timing.format_discord_message(report)
    assert "데이터오류" in msg


# ── 2축 레짐: posture 기반 multiplier (2026-06-11) ──────

@pytest.mark.unit
def test_posture_multiplier_throttles_down_calm():
    assert dca_timing._posture_multiplier("throttle") == 0.5
    assert dca_timing._posture_multiplier("capitulation_buy") == 1.3
    assert dca_timing._posture_multiplier("normal") == 1.0
    assert dca_timing._posture_multiplier("accumulate") == 1.0


@pytest.mark.unit
def test_posture_multiplier_unknown_falls_back():
    assert dca_timing._posture_multiplier(None) is None
    assert dca_timing._posture_multiplier("weird") is None


@pytest.mark.unit
def test_load_posture_and_throttle_halves(monkeypatch, tmp_path):
    """throttle posture → DCA 점수 반감 경로 (정돈된 하락 신규 억제)."""
    import json as _json
    (tmp_path / "last_regime.json").write_text(
        _json.dumps({"label": "neutral", "posture": "throttle"}))
    monkeypatch.setattr(dca_timing, "STATE_DIR", tmp_path)
    assert dca_timing._load_posture() == "throttle"
    assert dca_timing._posture_multiplier(dca_timing._load_posture()) == 0.5
