"""Corvin Jarvis — Multi-Account Aggregation (Tier 3.1)

자산 = 주식 + IRP/ISA + 미국 IRA + 현금/예금 + 부동산 + 암호화폐.
accounts.json (사용자 작성) 읽고 type/liquidity별 통합.

⚠️ 현재 portfolio.json은 미국 주식 brokerage만. accounts.json은 폐하가 직접 작성/갱신.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.accounts")

_LIQUIDITY_ORDER = ["T+0", "T+1", "T+2", "T+3", "weeks", "months", "years", "illiquid"]


def load_accounts(path: Path) -> list[dict[str, Any]]:
    """accounts.json 로드. 누락/오류 시 빈 list."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        items = data.get("accounts", [])
        return list(items) if isinstance(items, list) else []
    except (json.JSONDecodeError, OSError) as e:
        log.warning("accounts load failed: %s", e)
        return []


def aggregate_by_type(accounts_data: list[dict[str, Any]]) -> dict[str, int]:
    """자산 type 별 KRW 합산."""
    out: dict[str, int] = {}
    for a in accounts_data:
        t = a.get("type", "unknown")
        v = int(a.get("value_krw", 0) or 0)
        out[t] = out.get(t, 0) + v
    return out


def aggregate_by_liquidity(accounts_data: list[dict[str, Any]]) -> dict[str, int]:
    """유동성 그룹별 KRW 합산. 고정 ordering."""
    grouped: dict[str, int] = {}
    for a in accounts_data:
        liq = a.get("liquidity", "unknown")
        v = int(a.get("value_krw", 0) or 0)
        grouped[liq] = grouped.get(liq, 0) + v
    # ordered re-build
    out: dict[str, int] = {}
    for key in _LIQUIDITY_ORDER:
        if key in grouped:
            out[key] = grouped.pop(key)
    for key, v in grouped.items():  # unknown keys at end
        out[key] = v
    return out


def total_net_worth(accounts_data: list[dict[str, Any]]) -> int:
    """모든 계좌의 value_krw 합."""
    return sum(int(a.get("value_krw", 0) or 0) for a in accounts_data)
