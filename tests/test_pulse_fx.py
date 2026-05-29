"""⑥ 실시간 환율 KRW 환산 — pulse.fetch_portfolio의 FX 합산 로직."""
from __future__ import annotations

import json

from corvin_jarvis import pulse


def test_krw_equivalent_uses_live_rate_when_available():
    val, rate, src = pulse._krw_equivalent(
        total_krw=1_000_000, total_usd=1000, live_rate=1501.0, assumed_rate=1497.15)
    assert val == 1_000_000 + 1000 * 1501.0   # 실시간 환율로 환산
    assert rate == 1501.0
    assert src == "live"


def test_krw_equivalent_falls_back_to_assumed_when_no_live():
    val, rate, src = pulse._krw_equivalent(
        total_krw=0, total_usd=1000, live_rate=None, assumed_rate=1497.15)
    assert val == 1000 * 1497.15
    assert rate == 1497.15
    assert src == "assumed"


def test_krw_equivalent_ignores_nonpositive_live_rate():
    _, rate, src = pulse._krw_equivalent(
        total_krw=0, total_usd=1000, live_rate=0.0, assumed_rate=1497.15)
    assert rate == 1497.15 and src == "assumed"   # 0/음수 라이브값은 신뢰 불가 → 폴백


def test_krw_equivalent_none_when_no_rate_at_all():
    val, rate, src = pulse._krw_equivalent(
        total_krw=0, total_usd=1000, live_rate=None, assumed_rate=None)
    assert val is None and rate is None and src == "none"


def test_fetch_portfolio_populates_live_fx_equiv(tmp_path, monkeypatch):
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({
        "holdings": [
            {"symbol": "MSFT", "shares": 2, "avgPriceUSD": 400.0, "currency": "USD"},
            {"symbol": "012450", "shares": 4, "avgPriceKRW": 1_200_000, "currency": "KRW"},
        ],
        "totals": {"fxAssumedUSDKRW": 1497.15},
    }))
    monkeypatch.setattr(pulse, "PORTFOLIO_FILE", pf)

    class _Q:
        def __init__(self, price):
            self.price, self.pct_change, self.source, self.error = price, 0.0, "stub", None

    prices = {"MSFT": 500.0, "012450": 1_300_000.0}
    monkeypatch.setattr(pulse.quote_provider, "get_stock_quote", lambda s: _Q(prices[s]))

    _, summary = pulse.fetch_portfolio(usd_krw_rate=1501.0)
    assert summary["total_value_usd"] == 1000.0          # MSFT 500*2
    assert summary["total_value_krw"] == 5_200_000.0     # 012450 1.3M*4
    assert summary["fx_rate_used"] == 1501.0
    assert summary["fx_rate_source"] == "live"
    assert summary["total_value_krw_equiv"] == 5_200_000.0 + 1000.0 * 1501.0
