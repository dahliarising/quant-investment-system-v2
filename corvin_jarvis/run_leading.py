"""Corvin 선행 인텔리전스 — cron 진입점.

4 Pillar provider를 수집 → 게이트 → 브리프 → Telegram 발송.
실제 라이브 fetch provider 배선은 추후(Plan 5+); 현재는 이벤트 캘린더만 라이브.

⚠️ crontab 자동 등록 금지(정책). 사용자 수동 등록:
    30 8 * * 1-5 cd <repo> && python3 -m corvin_jarvis.run_leading >> state/leading.log 2>&1
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from corvin_jarvis import channels, leading_orchestrator as orch
from corvin_jarvis.signals import event_calendar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("corvin.run_leading")


def _today() -> date:
    """오늘 날짜 (cron 환경 결정성 위해 분리 — 테스트서 monkeypatch 가능)."""
    return date.today()


def main() -> int:
    as_of = _today()
    providers = [
        lambda: event_calendar.build_event_signals(
            as_of=as_of, earnings_rows=[], macro_horizon_days=30,
        ),
    ]
    signals = orch.collect(providers)
    brief = orch.format_brief(signals, threshold=60.0)
    sent = orch.dispatch_brief(brief, sender=channels.send_telegram)
    log.info("leading brief: signals=%d sent=%s", len(signals), sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
