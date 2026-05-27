"""Portfolio staleness detector.

portfolio.json이 며칠 지났는지 검사하고 severity 산정.
전략 생성 전 의무적으로 호출되어 stale 시 사용자에게 갱신 요청 트리거.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
PORTFOLIO_FILE: Final = PROJECT_ROOT / "portfolio.json"

FRESH_DAYS: Final = 3
WARN_DAYS: Final = 7
STALE_DAYS: Final = 14
CRITICAL_DAYS: Final = 30


class Severity(str, Enum):
    FRESH = "fresh"
    WARN = "warn"
    STALE = "stale"
    CRITICAL = "critical"


@dataclass(frozen=True)
class StalenessReport:
    severity: Severity
    days_since_update: int
    updated_at: str
    holdings_count: int
    block_strategy: bool
    user_prompt: str
    discord_banner: str


def _days_since(date_str: str) -> int:
    parsed = datetime.fromisoformat(date_str).date()
    return (date.today() - parsed).days


def check() -> StalenessReport:
    """포트폴리오 stale 여부 체크."""
    if not PORTFOLIO_FILE.exists():
        return StalenessReport(
            severity=Severity.CRITICAL,
            days_since_update=-1,
            updated_at="MISSING",
            holdings_count=0,
            block_strategy=True,
            user_prompt="portfolio.json 파일이 없습니다. 증권사 잔고 스크린샷 첨부해주십시오.",
            discord_banner="🚨 **CRITICAL** — portfolio.json 미존재. 전략 생성 차단.",
        )

    data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    updated_at = data.get("updatedAt", "1970-01-01")
    holdings = data.get("holdings", [])
    days = _days_since(updated_at)

    if days >= CRITICAL_DAYS:
        sev, block = Severity.CRITICAL, True
        prompt = (
            f"포트폴리오 갱신 {days}일 경과 — 매매가 있었을 가능성 매우 높음. "
            "전략 생성 전 증권사 잔고 스크린샷 필수."
        )
        banner = (
            f"🚨 **CRITICAL STALENESS** — portfolio.json {days}일 미갱신.\n"
            f"전략 생성 **차단**. 스크린샷 받기 전까지 진행 불가."
        )
    elif days >= STALE_DAYS:
        sev, block = Severity.STALE, True
        prompt = (
            f"포트폴리오 {days}일 경과 — 최신 잔고 스크린샷 보내주십시오. "
            "갱신 전까지 holdings 가정 전략 금지."
        )
        banner = (
            f"⚠️ **STALE** — portfolio.json {days}일 경과 (임계 {STALE_DAYS}일).\n"
            f"전략 생성 차단 — 잔고 스크린샷 필요."
        )
    elif days >= WARN_DAYS:
        sev, block = Severity.WARN, False
        prompt = (
            f"포트폴리오 {days}일 경과 — 매매 변경 있었으면 스크린샷 갱신 권장."
        )
        banner = (
            f"🟡 **WARN** — portfolio.json {days}일 경과. "
            f"매매 변경 있으면 갱신 부탁드립니다 (자동차단 임계 {STALE_DAYS}일)."
        )
    else:
        sev, block = Severity.FRESH, False
        prompt = ""
        banner = f"✅ FRESH — portfolio.json {days}일 경과 (보유 {len(holdings)}종목)"

    return StalenessReport(
        severity=sev,
        days_since_update=days,
        updated_at=updated_at,
        holdings_count=len(holdings),
        block_strategy=block,
        user_prompt=prompt,
        discord_banner=banner,
    )


def assert_fresh_for_strategy() -> StalenessReport:
    """전략 생성 직전 호출. STALE 이상이면 caller가 차단해야 함.

    Returns:
        StalenessReport with block_strategy flag.
    """
    return check()


if __name__ == "__main__":
    report = check()
    print(report.discord_banner)
    print(f"days={report.days_since_update} severity={report.severity.value} block={report.block_strategy}")
    if report.user_prompt:
        print(f"prompt: {report.user_prompt}")
