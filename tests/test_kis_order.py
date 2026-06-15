"""Tests for corvin_jarvis.kis_order — KIS 모의 주문 모듈 (mocked HTTP).

주문은 시세와 달리 POST + hashkey 헤더 필요 + 모의→실전 가드 필수.
모든 테스트는 requests.post/get을 mock — 실제 주문 절대 안 나감.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from corvin_jarvis import kis_auth, kis_order


@pytest.fixture
def mock_env() -> kis_auth.KISEnv:
    return kis_auth.KISEnv(
        app_key="k", app_secret="s",
        base_url=kis_auth.MOCK_URL, env="mock",
        account_no="12345678-01",
    )


@pytest.fixture
def prod_env() -> kis_auth.KISEnv:
    return kis_auth.KISEnv(
        app_key="k", app_secret="s",
        base_url=kis_auth.PROD_URL, env="prod",
        account_no="12345678-01",
    )


@pytest.fixture(autouse=True)
def _patch_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_headers → 토큰 발급(real HTTP) 차단."""
    monkeypatch.setattr(
        kis_auth, "build_headers",
        lambda env, tr_id: {"tr_id": tr_id, "authorization": "Bearer T",
                            "appkey": env.app_key, "appsecret": env.app_secret},
    )


@pytest.fixture(autouse=True)
def _zero_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kis_order.time, "sleep", lambda _s: None)


def _resp(payload: dict[str, Any], status: int = 200) -> Any:
    class _R:
        status_code = status
        text = ""
        def json(self) -> dict[str, Any]:
            return payload
    return _R()


def _hashkey_resp() -> Any:
    return _resp({"HASH": "HASHED123"})


def _order_ok_resp() -> Any:
    return _resp({"rt_cd": "0", "msg1": "정상처리", "output": {"ODNO": "0001234567", "ORD_TMD": "100000"}})


# ============================================================
# TR_ID 환경 분기 (모의=V, 실전=T)
# ============================================================

@pytest.mark.unit
def test_resolve_tr_id_mock_uses_v_prefix(mock_env: kis_auth.KISEnv) -> None:
    assert kis_order._resolve_tr_id(kis_order.TR_KR_BUY, mock_env).startswith("V")


@pytest.mark.unit
def test_resolve_tr_id_prod_uses_t_prefix(prod_env: kis_auth.KISEnv) -> None:
    assert kis_order._resolve_tr_id(kis_order.TR_KR_BUY, prod_env).startswith("T")


# ============================================================
# 계좌번호 분리
# ============================================================

@pytest.mark.unit
def test_split_account_on_hyphen() -> None:
    cano, prdt = kis_order._split_account("12345678-01")
    assert cano == "12345678"
    assert prdt == "01"


# ============================================================
# 🛡️ 모의→실전 가드 (가장 중요 — 약속: 실전 안 건드림)
# ============================================================

@pytest.mark.unit
def test_prod_order_blocked_without_confirm(prod_env: kis_auth.KISEnv) -> None:
    """실전 env인데 confirm_live=False → HTTP 호출 0, 차단된 OrderResult."""
    with patch("corvin_jarvis.kis_order.requests.post") as mock_post:
        res = kis_order.place_kr_order(
            "005930", side="buy", qty=1, price=70000, env=prod_env,
        )
    assert res.ok is False
    assert "실전" in res.error or "confirm_live" in res.error
    mock_post.assert_not_called()  # 주문 한 발도 안 나감


# ============================================================
# 입력 검증
# ============================================================

@pytest.mark.unit
def test_rejects_nonpositive_qty(mock_env: kis_auth.KISEnv) -> None:
    with patch("corvin_jarvis.kis_order.requests.post") as mock_post:
        res = kis_order.place_kr_order("005930", side="buy", qty=0, price=70000, env=mock_env)
    assert res.ok is False
    mock_post.assert_not_called()


@pytest.mark.unit
def test_rejects_unknown_side(mock_env: kis_auth.KISEnv) -> None:
    with patch("corvin_jarvis.kis_order.requests.post") as mock_post:
        res = kis_order.place_kr_order("005930", side="hodl", qty=1, price=70000, env=mock_env)
    assert res.ok is False
    mock_post.assert_not_called()


# ============================================================
# 매수/매도 주문 — body 구성 + TR_ID + 결과 파싱
# ============================================================

@pytest.mark.unit
def test_buy_order_builds_body_and_returns_order_no(mock_env: kis_auth.KISEnv) -> None:
    with patch(
        "corvin_jarvis.kis_order.requests.post",
        side_effect=[_hashkey_resp(), _order_ok_resp()],
    ) as mock_post:
        res = kis_order.place_kr_order(
            "005930", side="buy", qty=3, price=70000, env=mock_env,
        )
    assert res.ok is True
    assert res.order_no == "0001234567"
    # 두 번째 호출(주문)의 body 검증
    order_call = mock_post.call_args_list[1]
    body = order_call.kwargs["json"]
    assert body["PDNO"] == "005930"
    assert body["ORD_QTY"] == "3"
    assert body["ORD_UNPR"] == "70000"
    assert body["CANO"] == "12345678"
    assert body["ACNT_PRDT_CD"] == "01"
    assert body["ORD_DVSN"] == "00"  # 지정가
    # 주문 호출 tr_id = 매수(V로 시작)
    headers = order_call.kwargs["headers"]
    assert headers["tr_id"] == "VTTC0802U"
    assert headers["hashkey"] == "HASHED123"


@pytest.mark.unit
def test_sell_order_uses_sell_tr_id(mock_env: kis_auth.KISEnv) -> None:
    with patch(
        "corvin_jarvis.kis_order.requests.post",
        side_effect=[_hashkey_resp(), _order_ok_resp()],
    ) as mock_post:
        res = kis_order.place_kr_order("005930", side="sell", qty=1, price=70000, env=mock_env)
    assert res.ok is True
    assert mock_post.call_args_list[1].kwargs["headers"]["tr_id"] == "VTTC0801U"


@pytest.mark.unit
def test_market_order_sets_dvsn_01_and_zero_price(mock_env: kis_auth.KISEnv) -> None:
    with patch(
        "corvin_jarvis.kis_order.requests.post",
        side_effect=[_hashkey_resp(), _order_ok_resp()],
    ) as mock_post:
        res = kis_order.place_kr_order(
            "005930", side="buy", qty=2, price=None, order_type="market", env=mock_env,
        )
    assert res.ok is True
    body = mock_post.call_args_list[1].kwargs["json"]
    assert body["ORD_DVSN"] == "01"   # 시장가
    assert body["ORD_UNPR"] == "0"


@pytest.mark.unit
def test_order_rejection_returns_not_ok(mock_env: kis_auth.KISEnv) -> None:
    """rt_cd != 0 → ok=False + 거부 메시지 보존."""
    reject = _resp({"rt_cd": "1", "msg1": "주문가능금액 부족", "output": {}})
    with patch(
        "corvin_jarvis.kis_order.requests.post",
        side_effect=[_hashkey_resp(), reject],
    ):
        res = kis_order.place_kr_order("005930", side="buy", qty=999, price=70000, env=mock_env)
    assert res.ok is False
    assert "부족" in res.error


@pytest.mark.unit
def test_limit_order_requires_price(mock_env: kis_auth.KISEnv) -> None:
    """지정가인데 price 없음 → 검증 실패, 주문 안 나감."""
    with patch("corvin_jarvis.kis_order.requests.post") as mock_post:
        res = kis_order.place_kr_order("005930", side="buy", qty=1, price=None, env=mock_env)
    assert res.ok is False
    mock_post.assert_not_called()


# ============================================================
# 잔고 조회 (주문가능현금)
# ============================================================

@pytest.mark.unit
def test_get_kr_balance_parses_ord_psbl_cash(mock_env: kis_auth.KISEnv) -> None:
    payload = {
        "rt_cd": "0",
        "output1": [{"pdno": "005930", "hldg_qty": "3", "prpr": "70000"}],
        "output2": [{"ord_psbl_cash": "1500000", "tot_evlu_amt": "5000000"}],
    }
    with patch("corvin_jarvis.kis_order.requests.get", return_value=_resp(payload)):
        bal = kis_order.get_kr_balance(env=mock_env)
    assert bal.orderable_cash == 1500000.0
    assert bal.holdings[0]["pdno"] == "005930"
