"""Tests for corvin_jarvis.quote_provider — unified KIS+yfinance adapter."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from corvin_jarvis import kis_auth, kis_quote, quote_provider


@pytest.fixture(autouse=True)
def _reset_provider_caches() -> None:
    quote_provider.reset_caches()


# ============================================================
# Symbol classification
# ============================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "symbol,fn,expected",
    [
        ("005930", quote_provider.is_kr_stock, True),
        ("NVDA", quote_provider.is_kr_stock, False),
        ("12345", quote_provider.is_kr_stock, False),
        ("^GSPC", quote_provider.is_us_index, True),
        ("NVDA", quote_provider.is_us_index, False),
        ("KS11", quote_provider.is_kr_index, True),
        ("KQ11", quote_provider.is_kr_index, True),
        ("ks11", quote_provider.is_kr_index, True),
        ("BZ=F", quote_provider.is_commodity, True),
        ("KRW=X", quote_provider.is_fx, True),
        ("DX-Y.NYB", quote_provider.is_fx, True),
        ("NVDA", quote_provider.is_fx, False),
    ],
)
def test_symbol_classification(symbol: str, fn: Any, expected: bool) -> None:
    assert fn(symbol) is expected


# ============================================================
# KR stock — KIS primary
# ============================================================


@pytest.mark.unit
def test_kr_stock_quote_uses_kis_when_env_present() -> None:
    fake_env = MagicMock(env="mock")
    with patch.object(kis_auth, "load_env", return_value=fake_env), \
         patch.object(kis_quote, "get_kr_quote", return_value=(73500.0, 1.25)):
        q = quote_provider.get_stock_quote("005930")
    assert q.price == 73500.0
    assert q.pct_change == 1.25
    assert q.source == "kis_live"


@pytest.mark.unit
def test_kr_stock_quote_falls_back_to_pykrx_when_no_kis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kis_auth, "load_env",
        MagicMock(side_effect=kis_auth.KISConfigError("no key")),
    )
    fake_kr_data = MagicMock()
    fake_kr_data.get_kr_stock_data.return_value = {
        "현재가": 73000, "등락률(%)": 0.5,
    }
    with patch.dict("sys.modules", {"kr_data": fake_kr_data}):
        q = quote_provider.get_stock_quote("005930")
    assert q.price == 73000.0
    assert q.pct_change == 0.5
    assert q.source == "pykrx"


@pytest.mark.unit
def test_kr_stock_quote_falls_back_when_kis_returns_none() -> None:
    """KIS 정상 호출이지만 (None, None) 반환 시도 폴백."""
    fake_env = MagicMock(env="mock")
    fake_kr_data = MagicMock()
    fake_kr_data.get_kr_stock_data.return_value = {
        "현재가": 50000, "등락률(%)": -1.0,
    }
    with patch.object(kis_auth, "load_env", return_value=fake_env), \
         patch.object(kis_quote, "get_kr_quote", return_value=(None, None)), \
         patch.dict("sys.modules", {"kr_data": fake_kr_data}):
        q = quote_provider.get_stock_quote("000999")
    assert q.price == 50000.0
    assert q.source == "pykrx"


# ============================================================
# US stock — KIS primary, yfinance fallback
# ============================================================


@pytest.mark.unit
def test_us_stock_quote_uses_kis_with_exchange_from_universe() -> None:
    fake_env = MagicMock(env="mock")
    quote_provider._universe_exchange_map_cache = {"AAPL": "NAS"}
    with patch.object(kis_auth, "load_env", return_value=fake_env), \
         patch.object(kis_quote, "get_us_quote", return_value=(175.50, 0.69)) as m:
        q = quote_provider.get_stock_quote("AAPL")
    assert q.price == 175.50
    assert q.pct_change == 0.69
    assert q.source == "kis_live"
    _, kwargs = m.call_args
    assert kwargs.get("exchange") == "NAS"


@pytest.mark.unit
def test_us_stock_falls_back_to_yfinance_when_no_kis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kis_auth, "load_env",
        MagicMock(side_effect=kis_auth.KISConfigError("missing")),
    )

    class _Series:
        def __init__(self, vals: list[float]) -> None:
            self._vals = vals

        @property
        def iloc(self) -> Any:
            return self._vals

    class _DF:
        empty = False

        def __getitem__(self, k: str) -> _Series:
            return _Series([100.0, 102.0])

        def __len__(self) -> int:
            return 2

    fake_yf = MagicMock()
    fake_yf.Ticker.return_value.history.return_value = _DF()
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        q = quote_provider.get_stock_quote("NVDA")
    assert q.price == 102.0
    assert q.pct_change == pytest.approx(2.0, abs=0.01)
    assert q.source == "yfinance"


# ============================================================
# generic dispatcher routing
# ============================================================


@pytest.mark.unit
def test_get_quote_routes_kr_index_to_fdr() -> None:
    fake_kr_data = MagicMock()
    fake_kr_data.get_kr_index_data.return_value = {
        "현재가": 2500.0, "전일대비(%)": -0.5,
    }
    with patch.dict("sys.modules", {"kr_data": fake_kr_data}):
        q = quote_provider.get_quote("KS11")
    assert q.price == 2500.0
    assert q.source == "fdr"
    fake_kr_data.get_kr_index_data.assert_called_once_with("KS11")


@pytest.mark.unit
def test_get_quote_routes_commodity_to_yfinance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """원자재는 항상 yfinance — KIS env 무관."""
    monkeypatch.setattr(
        kis_auth, "load_env",
        MagicMock(side_effect=kis_auth.KISConfigError("none")),
    )

    class _S:
        @property
        def iloc(self) -> Any:
            return [80.0, 82.0]

    class _DF:
        empty = False

        def __getitem__(self, k: str) -> _S:
            return _S()

        def __len__(self) -> int:
            return 2

    fake_yf = MagicMock()
    fake_yf.Ticker.return_value.history.return_value = _DF()
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        q = quote_provider.get_quote("BZ=F")
    assert q.price == 82.0
    assert q.source == "yfinance"


# ============================================================
# daily closes routing
# ============================================================


@pytest.mark.unit
def test_get_stock_daily_closes_kis_first() -> None:
    fake_env = MagicMock(env="mock")
    with patch.object(kis_auth, "load_env", return_value=fake_env), \
         patch.object(kis_quote, "get_kr_daily_closes",
                      return_value=[100.0, 101.0, 102.0]):
        closes = quote_provider.get_stock_daily_closes("005930", days=10)
    assert closes == [100.0, 101.0, 102.0]


@pytest.mark.unit
def test_get_stock_daily_closes_kis_empty_falls_back() -> None:
    """KIS 빈 리스트 반환 시 yfinance fallback (US)."""
    fake_env = MagicMock(env="mock")
    quote_provider._universe_exchange_map_cache = {}

    class _S:
        def tolist(self) -> list[float]:
            return [200.0, 201.0]

        def tail(self, n: int) -> "_S":
            return self

    class _DF:
        empty = False

        def __getitem__(self, k: str) -> _S:
            return _S()

    fake_yf = MagicMock()
    fake_yf.Ticker.return_value.history.return_value = _DF()
    with patch.object(kis_auth, "load_env", return_value=fake_env), \
         patch.object(kis_quote, "get_us_daily_closes", return_value=[]), \
         patch.dict("sys.modules", {"yfinance": fake_yf}):
        closes = quote_provider.get_stock_daily_closes("NVDA", days=5)
    assert closes == [200.0, 201.0]


# ============================================================
# Quote.to_dict (pulse.py 호환 — dict serialization)
# ============================================================


@pytest.mark.unit
def test_quote_to_dict_has_pulse_shape() -> None:
    q = quote_provider.Quote(price=100.0, pct_change=1.5, source="kis_live")
    d = q.to_dict()
    assert d == {"price": 100.0, "pct_change": 1.5, "source": "kis_live", "error": None}


# ============================================================
# Fundamentals routing
# ============================================================


@pytest.mark.unit
def test_fundamentals_kr_uses_pykrx() -> None:
    fake_kr_data = MagicMock()
    fake_kr_data.get_kr_stock_data.return_value = {
        "PER": 12.3, "PBR": 1.4, "시가총액": 1234567890,
    }
    with patch.dict("sys.modules", {"kr_data": fake_kr_data}):
        out = quote_provider.get_stock_fundamentals("005930")
    assert out["PER"] == 12.3
    fake_kr_data.get_kr_stock_data.assert_called_once_with("005930")


@pytest.mark.unit
def test_fundamentals_us_uses_yfinance() -> None:
    fake_yf = MagicMock()
    fake_yf.Ticker.return_value.info = {
        "trailingPE": 25.0, "priceToBook": 8.0,
        "marketCap": 3e12, "dividendYield": 0.005,
    }
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        out = quote_provider.get_stock_fundamentals("NVDA")
    assert out["PER"] == 25.0
    assert out["_source"] == "yfinance"


# ============================================================
# Completed-bar filtering — exclude still-in-progress session
# (daily DCA must not score on a half-formed intraday bar)
# ============================================================

from datetime import date as _date, datetime as _datetime  # noqa: E402
from zoneinfo import ZoneInfo as _ZoneInfo  # noqa: E402

_NY = _ZoneInfo("America/New_York")


@pytest.mark.unit
def test_last_bar_incomplete_kr_during_session() -> None:
    """KR 종목 + 오늘자 봉 + 장중(15:30 이전) → 미완성(True)."""
    now = _datetime(2026, 5, 26, 11, 8, tzinfo=quote_provider.KST)
    assert quote_provider._last_bar_incomplete("012450", _date(2026, 5, 26), now=now) is True


@pytest.mark.unit
def test_last_bar_complete_kr_after_close() -> None:
    """KR 종목 + 오늘자 봉 + 마감(15:30) 이후 → 완성(False)."""
    now = _datetime(2026, 5, 26, 15, 40, tzinfo=quote_provider.KST)
    assert quote_provider._last_bar_incomplete("012450", _date(2026, 5, 26), now=now) is False


@pytest.mark.unit
def test_last_bar_complete_kr_past_date() -> None:
    """과거 거래일 봉 → 장중이어도 완성(False)."""
    now = _datetime(2026, 5, 26, 11, 8, tzinfo=quote_provider.KST)
    assert quote_provider._last_bar_incomplete("012450", _date(2026, 5, 22), now=now) is False


@pytest.mark.unit
def test_last_bar_incomplete_us_during_session() -> None:
    """US 종목 + ET 장중(09:30-16:00) → 미완성(True)."""
    now = _datetime(2026, 5, 26, 14, 0, tzinfo=_NY)
    assert quote_provider._last_bar_incomplete("NVDA", _date(2026, 5, 26), now=now) is True


@pytest.mark.unit
def test_last_bar_complete_us_after_close() -> None:
    """US 종목 + ET 마감(16:00) 이후 → 완성(False)."""
    now = _datetime(2026, 5, 26, 16, 30, tzinfo=_NY)
    assert quote_provider._last_bar_incomplete("NVDA", _date(2026, 5, 26), now=now) is False
