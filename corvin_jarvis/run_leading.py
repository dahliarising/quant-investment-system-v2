"""Corvin 선행 인텔리전스 — cron 진입점.

4 Pillar provider를 라이브 데이터로 수집 → 게이트 → 브리프 → Telegram 발송.

⚠️ crontab 자동 등록 금지(정책). 사용자 수동 등록:
    30 8 * * 1-5 cd <repo> && python3 -m corvin_jarvis.run_leading >> state/leading.log 2>&1
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from corvin_jarvis import (
    channels,
    dca_timing,
    leading_orchestrator as orch,
    leading_providers as lp,
    narrative,
    quote_provider,
)
from corvin_jarvis.signals import event_calendar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("corvin.run_leading")

# 보유/관심 종목 (portfolio + universe 핵심). 추후 universe.json 로딩으로 확장.
KR_SYMBOLS = ["012450"]
US_SYMBOLS = ["META", "MSFT", "NVDA", "TSLA"]
# 교차시장 타깃: (대상, 프록시명, 프록시심볼)
CROSS_TARGETS = [
    ("012450", "ITA 방산ETF 야간", "ITA"),
    ("META", "NQ 나스닥선물", "QQQ"),
]
# KOSPI 벤치마크: KODEX 200 ETF(069500) — KOSPI 지수는 FDR 전용이라 KIS 경로 미지원.
# KODEX200은 KR 종목이라 default_fetcher(KIS)로 정상 fetch + KOSPI200 추종.
KOSPI_BENCH_SYMBOL = "069500"


def _today() -> date:
    return date.today()


def _kr_bench_fetcher() -> list[float]:
    try:
        return dca_timing.default_fetcher(KOSPI_BENCH_SYMBOL, 252)
    except Exception as e:  # noqa: BLE001
        log.warning("KOSPI 벤치(069500) fetch 실패: %s", e)
        return []


def _proxy_pct_fetcher(proxy_symbol: str) -> float | None:
    try:
        q = quote_provider.get_quote(proxy_symbol)
        return q.pct_change if q and q.error is None else None
    except Exception as e:  # noqa: BLE001
        log.warning("프록시 %s fetch 실패: %s", proxy_symbol, e)
        return None


def _tone_fetcher(_sym: str) -> float | None:
    # narrative는 KR 시장 레벨 sentiment_tone만 제공(종목별 미지원)
    try:
        sig = narrative.latest_signal(narrative.DEFAULT_SIGNALS_DB, market="KR")
        return sig.get("sentiment_tone") if sig else None
    except Exception as e:  # noqa: BLE001
        log.warning("sentiment fetch 실패: %s", e)
        return None


def _metrics_fetcher(_sym: str) -> dict | None:
    # 재무 metrics 소스 배선은 후속(yfinance US financials / pykrx KR). 현재 미연결 → 스킵.
    return None


def main() -> int:
    as_of = _today()
    providers = [
        lambda: event_calendar.build_event_signals(
            as_of=as_of, earnings_rows=[], macro_horizon_days=30,
        ),
        lambda: lp.ensemble_provider(
            KR_SYMBOLS + US_SYMBOLS,
            price_fetcher=dca_timing.default_fetcher,
            bench_fetcher=_kr_bench_fetcher,
        ),
        lambda: lp.cross_market_provider(CROSS_TARGETS, pct_fetcher=_proxy_pct_fetcher),
        lambda: lp.fundamental_provider(
            KR_SYMBOLS + US_SYMBOLS,
            metrics_fetcher=_metrics_fetcher,
            tone_fetcher=_tone_fetcher,
        ),
    ]
    signals = orch.collect(providers)
    brief = orch.format_brief(signals, threshold=60.0)
    sent = orch.dispatch_brief(brief, sender=channels.send_telegram)
    log.info("leading brief: signals=%d sent=%s", len(signals), sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
