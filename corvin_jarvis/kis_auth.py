"""Corvin Jarvis — KIS (한국투자증권) OAuth (Tier 2.5)

KIS OpenAPI 토큰 발급 + 캐시 + 자동 갱신.

환경변수:
    KIS_APP_KEY      — 한국투자증권 발급 APP KEY
    KIS_APP_SECRET   — APP SECRET
    KIS_ENV          — "prod" 또는 "mock" (default: mock)
    KIS_ACCOUNT_NO   — 계좌번호 (시세 조회만 하면 더미값 OK, 주문 시 필요)

토큰 정책:
- 발급 자체 rate limit (분당 1회) → 캐시 필수
- TTL 24h. 만료 1h 전에 prefetch refresh
- state/kis_token.json에 캐시 저장 (PII 없음, AT만)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
TOKEN_CACHE = STATE_DIR / "kis_token.json"

PROD_URL = "https://openapi.koreainvestment.com:9443"
MOCK_URL = "https://openapivts.koreainvestment.com:29443"

REFRESH_MARGIN_SECONDS = 3600  # 만료 1h 전부터 재발급

log = logging.getLogger("corvin.kis.auth")


@dataclass(frozen=True)
class KISEnv:
    app_key: str
    app_secret: str
    base_url: str
    env: str  # "prod" | "mock"
    account_no: str = ""


class KISConfigError(RuntimeError):
    """KIS_APP_KEY/SECRET 환경변수 누락 등."""


def load_env() -> KISEnv:
    """환경변수에서 KIS 설정 로드. 누락 시 KISConfigError."""
    app_key = os.environ.get("KIS_APP_KEY", "").strip()
    app_secret = os.environ.get("KIS_APP_SECRET", "").strip()
    if not app_key or not app_secret:
        raise KISConfigError(
            "KIS_APP_KEY / KIS_APP_SECRET 환경변수 미설정 — "
            "https://apiportal.koreainvestment.com에서 발급 후 ~/.zshrc에 export"
        )
    env = os.environ.get("KIS_ENV", "mock").strip().lower()
    base_url = PROD_URL if env == "prod" else MOCK_URL
    return KISEnv(
        app_key=app_key,
        app_secret=app_secret,
        base_url=base_url,
        env=env,
        account_no=os.environ.get("KIS_ACCOUNT_NO", "").strip(),
    )


def _read_cache() -> dict[str, str]:
    if not TOKEN_CACHE.exists():
        return {}
    try:
        return json.loads(TOKEN_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_cache(data: dict[str, str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE.write_text(json.dumps(data, indent=2))


def _is_token_fresh(cache: dict[str, str], env_id: str) -> bool:
    if cache.get("env") != env_id:
        return False
    expires_at = cache.get("expires_at")
    token = cache.get("access_token")
    if not expires_at or not token:
        return False
    try:
        exp = datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    return exp - datetime.now(timezone.utc) > timedelta(seconds=REFRESH_MARGIN_SECONDS)


def _request_new_token(env: KISEnv) -> tuple[str, datetime]:
    """POST /oauth2/tokenP — 토큰 발급. expires_at(UTC) 함께 반환."""
    url = f"{env.base_url}/oauth2/tokenP"
    payload = {
        "grant_type": "client_credentials",
        "appkey": env.app_key,
        "appsecret": env.app_secret,
    }
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code != 200:
        raise KISConfigError(f"token 발급 실패 {r.status_code}: {r.text[:200]}")
    data = r.json()
    access_token = data.get("access_token")
    if not access_token:
        raise KISConfigError(f"access_token 없음: {data}")
    # expires_in seconds (보통 86400)
    expires_in = int(data.get("expires_in", 86400))
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    return access_token, expires_at


def get_token(env: KISEnv | None = None) -> str:
    """캐시된 토큰 반환. 없거나 만료 임박이면 신규 발급.

    KIS 토큰 발급은 분당 1회 제한 → 캐시 필수.
    """
    env = env or load_env()
    cache = _read_cache()
    if _is_token_fresh(cache, env.env):
        return cache["access_token"]

    log.info("KIS 토큰 발급/갱신 중 (env=%s)", env.env)
    token, expires_at = _request_new_token(env)
    _write_cache({
        "access_token": token,
        "expires_at": expires_at.isoformat(timespec="seconds"),
        "env": env.env,
        "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    log.info("KIS 토큰 발급 완료 (만료: %s)", expires_at.isoformat(timespec="minutes"))
    return token


def build_headers(env: KISEnv, tr_id: str) -> dict[str, str]:
    """공통 헤더 — authorization + appkey + appsecret + tr_id + custtype."""
    token = get_token(env)
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": env.app_key,
        "appsecret": env.app_secret,
        "tr_id": tr_id,
        "custtype": "P",  # P=개인
    }
