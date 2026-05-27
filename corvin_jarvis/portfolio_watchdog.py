"""Portfolio staleness watchdog — 주기적 자동 체크.

launchd로 매주 월요일 09:00 KST 실행. portfolio.json stale 시 Discord로 갱신 요청.
스킵 조건: severity == FRESH (조용히 종료).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "corvin_jarvis"))

from staleness import Severity, check  # noqa: E402

LOG_DIR: Final = PROJECT_ROOT / "corvin_jarvis" / "state"
LOG_DIR.mkdir(parents=True, exist_ok=True)
WATCHDOG_LOG: Final = LOG_DIR / "watchdog.log"
DISCORD_QUEUE: Final = LOG_DIR / "discord_pending.jsonl"

logging.basicConfig(
    filename=str(WATCHDOG_LOG),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("watchdog")


def enqueue_discord(text: str, severity: str) -> None:
    """Discord 전송 대기 큐에 적재. Corvin 세션이 다음 실행 시 송출."""
    payload = {
        "type": "portfolio_staleness",
        "severity": severity,
        "text": text,
        "channel": "stock-bot",
    }
    with DISCORD_QUEUE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log.info(f"Discord enqueued: severity={severity}")


def main() -> int:
    report = check()
    log.info(
        f"staleness: days={report.days_since_update} severity={report.severity.value} "
        f"holdings={report.holdings_count}"
    )

    if report.severity == Severity.FRESH:
        log.info("portfolio fresh — no action")
        return 0

    message = (
        f"📂 **Portfolio Watchdog — 주간 점검**\n\n"
        f"{report.discord_banner}\n\n"
        f"마지막 갱신: {report.updated_at} ({report.days_since_update}일 경과)\n"
        f"보유 종목: {report.holdings_count}개\n\n"
        f"**조치 요청**: {report.user_prompt}\n\n"
        f"_갱신 방법: 증권사 잔고 스크린샷 첨부 → Corvin이 자동 파싱_"
    )

    enqueue_discord(message, report.severity.value)
    print(message)
    return 0 if report.severity == Severity.WARN else 1


if __name__ == "__main__":
    sys.exit(main())
