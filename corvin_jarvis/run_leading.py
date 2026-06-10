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
    financial_metrics,
    leading_orchestrator as orch,
    leading_providers as lp,
    market_data,
    narrative,
    qualitative,
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

# 종목 사업 요약 (정성 분석 입력). universe.json theme 기반.
_SUMMARIES = {
    "012450": "한화에어로스페이스 — K-방산 글로벌 확산",
    "META": "Meta — AI 광고 + Reality Labs",
    "MSFT": "Microsoft — Azure + Copilot",
    "NVDA": "NVIDIA — AI GPU 절대강자",
    "TSLA": "Tesla — Optimus 휴머노이드",
}
def _today() -> date:
    return date.today()


def _qualitative_fetcher(sym: str) -> float | None:
    # Claude CLI(구독 인증) 사용 — ANTHROPIC_API_KEY 불필요. CLI 미설치면 None degrade.
    summary = _SUMMARIES.get(sym, sym)
    return qualitative.qualitative_score_via_cli(sym, summary)


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


def _metrics_fetcher(sym: str) -> dict | None:
    # US=yfinance financials, KR=pykrx 2시점 EPS. 데이터 없으면 None → 펀더멘털 스킵.
    return financial_metrics.fetch_metrics(sym)


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
            volume_fetcher=market_data.fetch_volume,
        ),
        lambda: lp.cross_market_provider(CROSS_TARGETS, pct_fetcher=_proxy_pct_fetcher),
        lambda: lp.fundamental_provider(
            KR_SYMBOLS + US_SYMBOLS,
            metrics_fetcher=_metrics_fetcher,
            tone_fetcher=_tone_fetcher,
            qualitative_fetcher=_qualitative_fetcher,
        ),
    ]
    signals = orch.collect(providers)
    try:
        from corvin_jarvis.signals import ledger
        ledger.record_batch("leading", [{
            "symbol": s.symbol, "kind": s.pillar, "direction": s.direction,
            "confidence": s.confidence,
            "horizon_days": ledger.horizon_str_to_days(s.horizon),
            "message": s.message, **s.evidence,
        } for s in signals])
    except Exception as e:  # noqa: BLE001 — 원장 실패는 신호 흐름 무영향
        log.warning("ledger record failed: %s", e)
    brief = orch.format_brief(signals, threshold=60.0)
    sent = orch.dispatch_brief(brief, sender=channels.send_telegram)
    log.info("leading brief: signals=%d sent=%s", len(signals), sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
