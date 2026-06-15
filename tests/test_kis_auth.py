"""Tests for corvin_jarvis.kis_auth — KIS OAuth + token cache."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from corvin_jarvis import kis_auth


@pytest.fixture
def kis_env() -> kis_auth.KISEnv:
    return kis_auth.KISEnv(
        app_key="testkey",
        app_secret="testsecret",
        base_url=kis_auth.MOCK_URL,
        env="mock",
        account_no="0000000000-01",
    )


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """env별 캐시 — mock 토큰은 kis_token_mock.json."""
    monkeypatch.setattr(kis_auth, "STATE_DIR", tmp_path)
    return tmp_path / "kis_token_mock.json"


@pytest.mark.unit
def test_load_env_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    with pytest.raises(kis_auth.KISConfigError):
        kis_auth.load_env()


@pytest.mark.unit
def test_load_env_mock_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIS_APP_KEY", "k")
    monkeypatch.setenv("KIS_APP_SECRET", "s")
    monkeypatch.delenv("KIS_ENV", raising=False)
    env = kis_auth.load_env()
    assert env.env == "mock"
    assert env.base_url == kis_auth.MOCK_URL


@pytest.mark.unit
def test_load_env_prod_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIS_APP_KEY", "k")
    monkeypatch.setenv("KIS_APP_SECRET", "s")
    monkeypatch.setenv("KIS_ENV", "prod")
    env = kis_auth.load_env()
    assert env.env == "prod"
    assert env.base_url == kis_auth.PROD_URL


@pytest.mark.unit
def test_is_token_fresh_returns_false_for_empty() -> None:
    assert kis_auth._is_token_fresh({}, "mock") is False


@pytest.mark.unit
def test_is_token_fresh_returns_false_when_env_mismatch() -> None:
    future = (datetime.now(timezone.utc) + timedelta(hours=23)).isoformat()
    cache = {"env": "prod", "expires_at": future, "access_token": "x"}
    assert kis_auth._is_token_fresh(cache, "mock") is False


@pytest.mark.unit
def test_is_token_fresh_true_when_well_within_ttl() -> None:
    future = (datetime.now(timezone.utc) + timedelta(hours=20)).isoformat()
    cache = {"env": "mock", "expires_at": future, "access_token": "x"}
    assert kis_auth._is_token_fresh(cache, "mock") is True


@pytest.mark.unit
def test_is_token_fresh_false_near_expiry() -> None:
    """만료 1h 미만이면 stale로 간주 → 재발급."""
    near = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    cache = {"env": "mock", "expires_at": near, "access_token": "x"}
    assert kis_auth._is_token_fresh(cache, "mock") is False


@pytest.mark.unit
def test_get_token_uses_fresh_cache(
    kis_env: kis_auth.KISEnv, isolated_cache: Path,
) -> None:
    future = (datetime.now(timezone.utc) + timedelta(hours=20)).isoformat()
    isolated_cache.write_text(json.dumps({
        "env": "mock", "expires_at": future, "access_token": "cached-token",
    }))
    with patch("corvin_jarvis.kis_auth._request_new_token") as mock_req:
        token = kis_auth.get_token(kis_env)
    assert token == "cached-token"
    mock_req.assert_not_called()


@pytest.mark.unit
def test_get_token_requests_new_when_no_cache(
    kis_env: kis_auth.KISEnv, isolated_cache: Path,
) -> None:
    fake_expires = datetime.now(timezone.utc) + timedelta(hours=24)
    with patch(
        "corvin_jarvis.kis_auth._request_new_token",
        return_value=("fresh-token", fake_expires),
    ) as mock_req:
        token = kis_auth.get_token(kis_env)
    assert token == "fresh-token"
    mock_req.assert_called_once_with(kis_env)
    # 캐시 파일 작성 확인
    cached = json.loads(isolated_cache.read_text())
    assert cached["access_token"] == "fresh-token"
    assert cached["env"] == "mock"


@pytest.mark.unit
def test_per_env_cache_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모의 토큰 발급이 실전 캐시를 덮어쓰지 않음 (별도 파일)."""
    monkeypatch.setattr(kis_auth, "STATE_DIR", tmp_path)
    future = (datetime.now(timezone.utc) + timedelta(hours=20)).isoformat()
    # 실전 캐시 미리 존재
    (tmp_path / "kis_token_prod.json").write_text(json.dumps({
        "env": "prod", "expires_at": future, "access_token": "PROD-TOKEN",
    }))
    mock_env = kis_auth.KISEnv("k", "s", kis_auth.MOCK_URL, "mock")
    fake_expires = datetime.now(timezone.utc) + timedelta(hours=24)
    with patch("corvin_jarvis.kis_auth._request_new_token",
               return_value=("MOCK-TOKEN", fake_expires)):
        kis_auth.get_token(mock_env)
    # 실전 캐시 보존 + 모의 캐시 별도 생성
    prod_cached = json.loads((tmp_path / "kis_token_prod.json").read_text())
    assert prod_cached["access_token"] == "PROD-TOKEN"
    mock_cached = json.loads((tmp_path / "kis_token_mock.json").read_text())
    assert mock_cached["access_token"] == "MOCK-TOKEN"


@pytest.mark.unit
def test_load_mock_env_reads_mock_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIS_MOCK_APP_KEY", "mk")
    monkeypatch.setenv("KIS_MOCK_APP_SECRET", "ms")
    monkeypatch.setenv("KIS_MOCK_ACCOUNT_NO", "50193344-01")
    env = kis_auth.load_mock_env()
    assert env.env == "mock"
    assert env.base_url == kis_auth.MOCK_URL
    assert env.app_key == "mk"
    assert env.account_no == "50193344-01"


@pytest.mark.unit
def test_load_mock_env_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KIS_MOCK_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_MOCK_APP_SECRET", raising=False)
    with pytest.raises(kis_auth.KISConfigError):
        kis_auth.load_mock_env()


@pytest.mark.unit
def test_request_new_token_parses_response(kis_env: kis_auth.KISEnv) -> None:
    fake_response = type("R", (), {
        "status_code": 200,
        "json": lambda self: {"access_token": "abc", "expires_in": 86400},
    })()
    with patch("corvin_jarvis.kis_auth.requests.post", return_value=fake_response):
        token, expires_at = kis_auth._request_new_token(kis_env)
    assert token == "abc"
    # expires_at은 거의 24h 미래
    delta = expires_at - datetime.now(timezone.utc)
    assert timedelta(hours=23) < delta < timedelta(hours=25)


@pytest.mark.unit
def test_request_new_token_raises_on_http_error(kis_env: kis_auth.KISEnv) -> None:
    fake_response = type("R", (), {
        "status_code": 401,
        "text": "invalid appkey",
    })()
    with patch("corvin_jarvis.kis_auth.requests.post", return_value=fake_response):
        with pytest.raises(kis_auth.KISConfigError):
            kis_auth._request_new_token(kis_env)


@pytest.mark.unit
def test_build_headers_includes_tr_id_and_bearer(
    kis_env: kis_auth.KISEnv, isolated_cache: Path,
) -> None:
    future = (datetime.now(timezone.utc) + timedelta(hours=20)).isoformat()
    isolated_cache.write_text(json.dumps({
        "env": "mock", "expires_at": future, "access_token": "T",
    }))
    hdrs = kis_auth.build_headers(kis_env, tr_id="FHKST01010100")
    assert hdrs["authorization"] == "Bearer T"
    assert hdrs["appkey"] == "testkey"
    assert hdrs["appsecret"] == "testsecret"
    assert hdrs["tr_id"] == "FHKST01010100"
    assert hdrs["custtype"] == "P"
