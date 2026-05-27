"""Tests for corvin_jarvis.kis_quote — KIS 시세 fetcher (mocked HTTP)."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from corvin_jarvis import kis_auth, kis_quote


@pytest.fixture
def kis_env() -> kis_auth.KISEnv:
    return kis_auth.KISEnv(
        app_key="k", app_secret="s",
        base_url=kis_auth.MOCK_URL, env="mock",
    )


@pytest.fixture(autouse=True)
def _patch_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_headers는 토큰 발급 시도 → 모든 테스트에서 stub."""
    monkeypatch.setattr(
        kis_auth, "build_headers",
        lambda env, tr_id: {"tr_id": tr_id, "authorization": "Bearer T"},
    )


@pytest.fixture(autouse=True)
def _zero_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kis_quote.time, "sleep", lambda _s: None)


def _make_response(payload: dict[str, Any], status: int = 200) -> Any:
    class _R:
        status_code = status
        text = ""
        def json(self) -> dict[str, Any]:
            return payload
    return _R()


def _rate_limit_response() -> Any:
    """EGW00201 초당 거래건수 초과 응답 시뮬레이션."""
    class _R:
        status_code = 500
        text = '{"rt_cd":"1","msg_cd":"EGW00201"}'
        def json(self) -> dict[str, Any]:
            return {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수 초과"}
    return _R()


# ============================================================
# 국내주식
# ============================================================


@pytest.mark.unit
def test_get_kr_current_price_parses_stck_prpr(kis_env: kis_auth.KISEnv) -> None:
    payload = {"rt_cd": "0", "output": {"stck_prpr": "73500"}}
    with patch("corvin_jarvis.kis_quote.requests.get", return_value=_make_response(payload)):
        price = kis_quote.get_kr_current_price("005930", env=kis_env)
    assert price == 73500.0


@pytest.mark.unit
def test_get_kr_current_price_returns_none_on_empty(kis_env: kis_auth.KISEnv) -> None:
    payload = {"rt_cd": "0", "output": {"stck_prpr": ""}}
    with patch("corvin_jarvis.kis_quote.requests.get", return_value=_make_response(payload)):
        assert kis_quote.get_kr_current_price("000000", env=kis_env) is None


@pytest.mark.unit
def test_get_kr_daily_closes_parses_output2(kis_env: kis_auth.KISEnv) -> None:
    payload = {
        "rt_cd": "0",
        "output2": [
            {"stck_bsop_date": "20260520", "stck_clpr": "73000"},
            {"stck_bsop_date": "20260521", "stck_clpr": "73500"},
            {"stck_bsop_date": "20260522", "stck_clpr": "74000"},
        ],
    }
    # second page empty → break
    with patch(
        "corvin_jarvis.kis_quote.requests.get",
        side_effect=[_make_response(payload), _make_response({"output2": []})],
    ):
        closes = kis_quote.get_kr_daily_closes("005930", days=10, env=kis_env)
    # oldest → newest
    assert closes == [73000.0, 73500.0, 74000.0]


@pytest.mark.unit
def test_get_kr_daily_closes_dedupes_overlapping_pages(
    kis_env: kis_auth.KISEnv,
) -> None:
    """페이지네이션 중복 날짜는 한 번만 카운트."""
    p1 = {"output2": [
        {"stck_bsop_date": "20260520", "stck_clpr": "100"},
        {"stck_bsop_date": "20260521", "stck_clpr": "101"},
    ]}
    p2 = {"output2": [
        {"stck_bsop_date": "20260520", "stck_clpr": "100"},  # 중복
        {"stck_bsop_date": "20260519", "stck_clpr": "99"},
    ]}
    with patch(
        "corvin_jarvis.kis_quote.requests.get",
        side_effect=[_make_response(p1), _make_response(p2), _make_response({"output2": []})],
    ):
        closes = kis_quote.get_kr_daily_closes("005930", days=200, env=kis_env)
    assert closes == [99.0, 100.0, 101.0]


# ============================================================
# 해외주식
# ============================================================


@pytest.mark.unit
def test_get_us_current_price_uses_last_field(kis_env: kis_auth.KISEnv) -> None:
    payload = {"output": {"last": "175.50"}}
    with patch("corvin_jarvis.kis_quote.requests.get", return_value=_make_response(payload)):
        price = kis_quote.get_us_current_price("AAPL", exchange="NAS", env=kis_env)
    assert price == 175.50


@pytest.mark.unit
def test_get_us_current_price_falls_back_to_nys_if_nas_empty(
    kis_env: kis_auth.KISEnv,
) -> None:
    empty = {"output": {"last": ""}}
    found = {"output": {"last": "100.00"}}
    with patch(
        "corvin_jarvis.kis_quote.requests.get",
        side_effect=[_make_response(empty), _make_response(found)],
    ):
        price = kis_quote.get_us_current_price("TSM", env=kis_env)  # no exchange hint
    assert price == 100.00


@pytest.mark.unit
def test_get_us_daily_closes_parses_xymd_clos(kis_env: kis_auth.KISEnv) -> None:
    payload = {
        "rt_cd": "0",
        "output2": [
            {"xymd": "20260520", "clos": "100.50"},
            {"xymd": "20260521", "clos": "101.25"},
        ],
    }
    with patch(
        "corvin_jarvis.kis_quote.requests.get",
        side_effect=[_make_response(payload), _make_response({"output2": []})],
    ):
        closes = kis_quote.get_us_daily_closes("AAPL", exchange="NAS", days=5, env=kis_env)
    assert closes == [100.50, 101.25]


@pytest.mark.unit
def test_get_us_daily_closes_returns_empty_when_excd_all_fail(
    kis_env: kis_auth.KISEnv,
) -> None:
    empty = {"output2": []}
    with patch("corvin_jarvis.kis_quote.requests.get", return_value=_make_response(empty)):
        closes = kis_quote.get_us_daily_closes("FAKE", days=10, env=kis_env)
    assert closes == []


# ============================================================
# make_fetcher (dca_timing 호환)
# ============================================================


@pytest.mark.unit
def test_make_fetcher_routes_kr_to_domestic(kis_env: kis_auth.KISEnv) -> None:
    fetcher = kis_quote.make_fetcher(env=kis_env)
    with patch(
        "corvin_jarvis.kis_quote.get_kr_daily_closes",
        return_value=[100.0, 101.0],
    ) as mock_kr, patch(
        "corvin_jarvis.kis_quote.get_us_daily_closes",
        return_value=[1.0],
    ) as mock_us:
        result = fetcher("005930", days=30)
    assert result == [100.0, 101.0]
    mock_kr.assert_called_once()
    mock_us.assert_not_called()


@pytest.mark.unit
def test_make_fetcher_routes_us_to_overseas_with_exchange_map(
    kis_env: kis_auth.KISEnv,
) -> None:
    fetcher = kis_quote.make_fetcher(
        exchange_map={"NVDA": "NAS"}, env=kis_env,
    )
    with patch(
        "corvin_jarvis.kis_quote.get_us_daily_closes",
        return_value=[200.0, 201.0],
    ) as mock_us:
        result = fetcher("NVDA", days=30)
    assert result == [200.0, 201.0]
    args, kwargs = mock_us.call_args
    assert kwargs.get("exchange") == "NAS"


@pytest.mark.unit
def test_make_fetcher_returns_empty_when_kis_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KIS env 없으면 빈 리스트 → dca_timing 쪽에서 fallback 가능."""
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    fetcher = kis_quote.make_fetcher(env=None)
    assert fetcher("NVDA", days=10) == []


# ============================================================
# Rate limit retry + backoff
# ============================================================


@pytest.mark.unit
def test_http_get_retries_on_rate_limit_then_succeeds(
    kis_env: kis_auth.KISEnv,
) -> None:
    """EGW00201 두 번 후 성공 응답 → 데이터 정상 반환."""
    success = _make_response({"rt_cd": "0", "output": {"stck_prpr": "100"}})
    with patch(
        "corvin_jarvis.kis_quote.requests.get",
        side_effect=[_rate_limit_response(), _rate_limit_response(), success],
    ) as mock_get:
        data = kis_quote._http_get(
            kis_env,
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            kis_quote.TR_DOMESTIC_PRICE,
            {"fid_input_iscd": "005930"},
        )
    assert data.get("rt_cd") == "0"
    assert mock_get.call_count == 3


@pytest.mark.unit
def test_http_get_returns_empty_after_max_retries(
    kis_env: kis_auth.KISEnv,
) -> None:
    """rate-limit이 retry 한도 초과 시 빈 dict."""
    with patch(
        "corvin_jarvis.kis_quote.requests.get",
        return_value=_rate_limit_response(),
    ) as mock_get:
        data = kis_quote._http_get(
            kis_env,
            "/uapi/overseas-price/v1/quotations/dailyprice",
            kis_quote.TR_OVERSEAS_DAILY,
            {"SYMB": "NVDA"},
        )
    assert data == {}
    assert mock_get.call_count == kis_quote.MAX_RETRIES_ON_RATE_LIMIT + 1


@pytest.mark.unit
def test_delay_overseas_longer_than_domestic() -> None:
    """해외 호출은 국내보다 더 긴 sleep."""
    assert (
        kis_quote._delay_for(kis_quote.TR_OVERSEAS_DAILY)
        > kis_quote._delay_for(kis_quote.TR_DOMESTIC_DAILY)
    )


@pytest.mark.unit
def test_is_rate_limit_error_detects_egw_codes() -> None:
    assert kis_quote._is_rate_limit_error({"msg_cd": "EGW00201"}) is True
    assert kis_quote._is_rate_limit_error({"msg_cd": "EGW00301"}) is True
    assert kis_quote._is_rate_limit_error({"msg_cd": "EGW00304"}) is False  # 인증 거절
    assert kis_quote._is_rate_limit_error({}) is False
