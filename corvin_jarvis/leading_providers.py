"""Corvin 선행 인텔리전스 — 라이브 데이터 provider 어댑터.

순수 점수 함수에 실제 데이터를 주입한다. 모든 provider는 fetcher 콜백을
주입받아(의존성 주입) 테스트 가능. 데이터 누락 시 해당 종목 스킵(추측 금지).
"""
from __future__ import annotations

from typing import Any, Callable

from corvin_jarvis.signals import cross_market, ensemble, fundamental
from corvin_jarvis.signals.leading_signal import LeadingSignal

# 앙상블 Minervini 최소 이력 (MA200 필요)
_MIN_ENSEMBLE_HISTORY = 200


def ensemble_provider(
    symbols: list[str],
    price_fetcher: Callable[..., list[float]],
    bench_fetcher: Callable[[], list[float]],
    volume_fetcher: Callable[[str], tuple[float | None, float | None]] | None = None,
) -> list[LeadingSignal]:
    """종목별 종가→지표→앙상블 신호. 이력 부족 종목은 스킵.

    volume_fetcher 주입 시 거래량 점수 반영, 없으면 NEUTRAL.
    """
    bench_closes = bench_fetcher()
    out: list[LeadingSignal] = []
    for sym in symbols:
        closes = price_fetcher(sym, 252)
        if len(closes) < _MIN_ENSEMBLE_HISTORY:
            continue
        ind = ensemble.indicators_from_closes(closes)
        minervini = ensemble.minervini_trend_score(
            price=ind["price"], ma50=ind["ma50"], ma150=ind["ma150"],
            ma200=ind["ma200"], low_52w=ind["low_52w"], high_52w=ind["high_52w"],
        )
        rs_windows = ensemble.rs_windows_from_closes(closes, bench_closes)
        rs = ensemble.multi_timeframe_rs_score(rs_windows)
        if volume_fetcher is not None:
            vol, vol_avg = volume_fetcher(sym)
        else:
            vol, vol_avg = None, None
        volume = ensemble.volume_breakthrough_score(vol, vol_avg)
        out.append(ensemble.build_ensemble_signal(sym, minervini, rs, volume))
    return out


def cross_market_provider(
    targets: list[tuple[str, str, str]],
    pct_fetcher: Callable[[str], float | None],
) -> list[LeadingSignal]:
    """targets: (대상종목, 프록시명, 프록시심볼). 프록시 pct 없으면 스킵."""
    out: list[LeadingSignal] = []
    for symbol, proxy_name, proxy_symbol in targets:
        pct = pct_fetcher(proxy_symbol)
        if pct is None:
            continue
        out.append(cross_market.build_cross_market_signal(symbol, proxy_name, pct))
    return out


def fundamental_provider(
    symbols: list[str],
    metrics_fetcher: Callable[[str], dict[str, Any] | None],
    tone_fetcher: Callable[[str], float | None],
    qualitative_fetcher: Callable[[str], float | None] | None = None,
) -> list[LeadingSignal]:
    """재무 metrics + 뉴스 tone (+선택 정성) → 펀더멘털 신호. 재무 없으면 스킵."""
    out: list[LeadingSignal] = []
    for sym in symbols:
        metrics = metrics_fetcher(sym)
        if not metrics:
            continue
        fin = fundamental.financial_growth_score(metrics)
        news = fundamental.news_sentiment_score(tone_fetcher(sym))
        qual = qualitative_fetcher(sym) if qualitative_fetcher else None
        out.append(fundamental.build_fundamental_signal(sym, fin, qual, news))
    return out
