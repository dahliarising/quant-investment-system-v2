"""Corvin Jarvis — Daily DCA Timing (Tier 2.4 add-on)

Value-tilted DCA candidate scoring.

매일 KST 08:00 (KR 시장) + 21:00 (US 시장) 실행:
1. universe.json의 33종목에 대해 yfinance/pykrx로 일봉 252일치 fetch
2. RSI / MA50 distance / 20d z-score / 52w drawdown 4종 신호 → 0~100 composite
3. Regime gate 곱: CRISIS=×1.3, RISK_OFF=×1.15, EUPHORIA=×0.7
4. score ≥ 50 (B2 Balanced) 통과 종목만 candidates
5. Value-tilted multiplier: 50-59 → 1x, 60-74 → 1.5x, 75+ → 2x
6. Tier-weighted 자본 배분: T1 50% · T2 35% · T3 15%
7. Discord push (`dca_signals` dedup namespace) + state/dca_signals.json 저장

⚠️ Advisory only — 모의 추천. 실제 매매 결정은 폐하 판단.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

try:
    from . import channels
except ImportError:  # script 실행 fallback
    import channels  # type: ignore[no-redef]

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
UNIVERSE_FILE = BASE_DIR / "universe.json"
CONFIG_FILE = BASE_DIR / "config.json"
SIGNALS_FILE = STATE_DIR / "dca_signals.json"
DEDUP_FILE = STATE_DIR / "dca_push_dedup.json"

KST = ZoneInfo("Asia/Seoul")
log = logging.getLogger("corvin.dca")

SCORE_THRESHOLD = 50  # B2 Balanced
MIN_HISTORY_DAYS = 30
TIER_WEIGHTS: dict[int, float] = {1: 0.50, 2: 0.35, 3: 0.15}
DEFAULT_DAILY_BUDGET_KRW = 50_000  # 폐하가 config로 override 가능

Fetcher = Callable[[str, int], list[float]]
LivePriceFetcher = Callable[[str], "float | None"]


# ============================================================
# Data classes
# ============================================================


@dataclass(frozen=True)
class TickerScore:
    symbol: str
    market: str  # "US" | "KR"
    tier: int
    score: int
    rsi: float | None
    ma50_distance_pct: float | None
    zscore_20d: float | None
    drawdown_52w_pct: float | None
    current_price: float  # 일봉 마지막 close
    multiplier: float
    allocation_krw: int
    breakdown: dict[str, int]
    live_price: float | None = None  # KIS 실시간 현재가 (사용 가능 시)
    live_delta_pct: float | None = None  # live vs daily close 차이 %


@dataclass(frozen=True)
class DCAReport:
    generated_at: str
    market_window: str  # "KR" | "US" | "ALL"
    regime: str | None
    regime_multiplier: float
    threshold: int
    daily_budget_krw: int
    candidates: list[TickerScore]
    skipped: list[dict[str, Any]] = field(default_factory=list)


# ============================================================
# Signal calculators
# ============================================================


def _rsi(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for prev, cur in zip(prices[-(period + 1) : -1], prices[-period:]):
        delta = cur - prev
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _ma_distance_pct(prices: list[float], window: int = 50) -> float | None:
    if len(prices) < window:
        return None
    ma = sum(prices[-window:]) / window
    if ma == 0:
        return None
    return (prices[-1] - ma) / ma * 100


def _zscore(prices: list[float], window: int = 20) -> float | None:
    if len(prices) < window:
        return None
    recent = prices[-window:]
    mu = statistics.fmean(recent)
    try:
        sigma = statistics.stdev(recent)
    except statistics.StatisticsError:
        return None
    if sigma == 0:
        return 0.0
    return (recent[-1] - mu) / sigma


def _drawdown_52w_pct(prices: list[float]) -> float | None:
    if len(prices) < MIN_HISTORY_DAYS:
        return None
    window = prices[-252:] if len(prices) >= 252 else prices
    high = max(window)
    if high == 0:
        return None
    return (prices[-1] - high) / high * 100


# ============================================================
# Composite & multipliers
# ============================================================


def _composite_score(
    rsi: float | None,
    ma_dist: float | None,
    z: float | None,
    dd_52w: float | None,
) -> tuple[int, dict[str, int]]:
    breakdown: dict[str, int] = {}

    if rsi is not None:
        if rsi <= 30:
            breakdown["rsi"] = 30
        elif rsi <= 40:
            breakdown["rsi"] = 15

    if ma_dist is not None:
        if ma_dist <= -10:
            breakdown["ma50"] = 30
        elif ma_dist <= -5:
            breakdown["ma50"] = 20

    if z is not None:
        if z <= -2:
            breakdown["zscore"] = 25
        elif z <= -1:
            breakdown["zscore"] = 15

    if dd_52w is not None and dd_52w <= -15:
        breakdown["drawdown_52w"] = 10

    return min(sum(breakdown.values()), 100), breakdown


def _score_to_multiplier(score: int) -> float:
    """Value-tilted DCA (option A2). Caller filters score < threshold."""
    if score >= 75:
        return 2.0
    if score >= 60:
        return 1.5
    return 1.0


def _regime_multiplier(regime: str | None) -> float:
    """regime.py는 lowercase ("crisis"/"risk_off"/...)로 반환 — case-insensitive.

    ⚠️ 단일축 fallback. 2축 posture가 있으면 _posture_multiplier 우선
    (risk_off가 무조건 가속하는 결함을 posture가 교정 — 정돈된 하락은 throttle).
    """
    key = (regime or "").upper()
    return {
        "CRISIS": 1.3,
        "RISK_OFF": 1.15,
        "RISK_ON": 1.0,
        "NEUTRAL": 1.0,
        "EUPHORIA": 0.7,
    }.get(key, 1.0)


# 2축 레짐 posture → DCA 신규 진입 multiplier (점수 배율)
_POSTURE_MULT = {
    "capitulation_buy": 1.3,   # 진짜 투매 — 역발상 가속
    "accumulate": 1.0,
    "normal": 1.0,
    "throttle": 0.5,           # 정돈된 하락(down+calm) — 신규 진입 억제(점수 반감)
}


def _posture_multiplier(posture: str | None) -> float | None:
    """posture → multiplier. 미지/None이면 None (라벨 multiplier로 fallback)."""
    return _POSTURE_MULT.get(posture) if posture in _POSTURE_MULT else None


def _load_posture() -> str | None:
    """jarvis 2축 레짐 posture — last_regime.json에서 (없으면 None)."""
    last = STATE_DIR / "last_regime.json"
    if last.exists():
        try:
            return json.loads(last.read_text()).get("posture")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _base_allocation_per_ticker(
    daily_budget_krw: int,
    tier: int,
    tier_counts: dict[int, int],
) -> int:
    n = tier_counts.get(tier, 0)
    weight = TIER_WEIGHTS.get(tier, 0.0)
    if n == 0:
        return 0
    return int(daily_budget_krw * weight / n)


def _is_kr_symbol(symbol: str) -> bool:
    return symbol.isdigit() and len(symbol) == 6


# ============================================================
# Daily-bar fetcher — quote_provider 위임 (KIS 우선 + yfinance/pykrx 폴백)
# ============================================================


def default_fetcher(symbol: str, days: int = 252) -> list[float]:
    """quote_provider에 위임 — single source of truth.

    completed_only=True: 일봉 전략이므로 진행 중인 오늘 봉을 제외해 완성봉만 평가
    (장중 수동 재실행 시에도 장 마감 후 스케줄 실행과 동일한 결과 보장).
    """
    from corvin_jarvis import quote_provider
    return quote_provider.get_stock_daily_closes(symbol, days=days, completed_only=True)


# ============================================================
# Core scoring + report building
# ============================================================


def _score_one(
    symbol: str,
    tier: int,
    daily_budget_krw: int,
    tier_counts: dict[int, int],
    regime_mult: float,
    fetcher: Fetcher,
    live_fetcher: LivePriceFetcher | None = None,
) -> tuple[TickerScore | None, str | None]:
    prices = fetcher(symbol, 252)
    if len(prices) < MIN_HISTORY_DAYS:
        return None, "insufficient_history"

    rsi = _rsi(prices)
    ma_dist = _ma_distance_pct(prices, window=50)
    z = _zscore(prices, window=20)
    dd_52w = _drawdown_52w_pct(prices)
    raw_score, breakdown = _composite_score(rsi, ma_dist, z, dd_52w)
    score = min(int(raw_score * regime_mult), 100)

    if score < SCORE_THRESHOLD:
        return None, f"below_threshold_score_{score}"

    mult = _score_to_multiplier(score)
    base = _base_allocation_per_ticker(daily_budget_krw, tier, tier_counts)
    alloc = int(base * mult)
    daily_close = round(prices[-1], 2)

    live_price: float | None = None
    live_delta_pct: float | None = None
    if live_fetcher is not None:
        try:
            live = live_fetcher(symbol)
        except Exception as e:  # noqa: BLE001 — 외부 API 실패 silent fallthrough
            log.debug("live price fetch failed for %s: %s", symbol, e)
            live = None
        if live is not None and live > 0 and daily_close > 0:
            live_price = round(live, 2)
            live_delta_pct = round((live - daily_close) / daily_close * 100, 2)

    return TickerScore(
        symbol=symbol,
        market="KR" if _is_kr_symbol(symbol) else "US",
        tier=tier,
        score=score,
        rsi=round(rsi, 2) if rsi is not None else None,
        ma50_distance_pct=round(ma_dist, 2) if ma_dist is not None else None,
        zscore_20d=round(z, 2) if z is not None else None,
        drawdown_52w_pct=round(dd_52w, 2) if dd_52w is not None else None,
        current_price=daily_close,
        multiplier=mult,
        allocation_krw=alloc,
        breakdown=breakdown,
        live_price=live_price,
        live_delta_pct=live_delta_pct,
    ), None


def build_dca_report(
    universe: list[dict[str, Any]],
    *,
    daily_budget_krw: int = DEFAULT_DAILY_BUDGET_KRW,
    market_window: str = "ALL",
    regime: str | None = None,
    fetcher: Fetcher | None = None,
    live_fetcher: LivePriceFetcher | None = None,
) -> DCAReport:
    """Universe scan → DCAReport.

    market_window: "KR" / "US" / "ALL" — D3 option은 외부에서 두 번 호출.
    live_fetcher: KIS 실시간 현재가 콜백 (None이면 daily close만 사용).
    """
    fetcher = fetcher or default_fetcher

    def in_window(sym: str) -> bool:
        if market_window == "ALL":
            return True
        is_kr = _is_kr_symbol(sym)
        return (market_window == "KR" and is_kr) or (market_window == "US" and not is_kr)

    active = [u for u in universe if in_window(u["symbol"])]
    tier_counts: dict[int, int] = {}
    for u in active:
        tier_counts[u["tier"]] = tier_counts.get(u["tier"], 0) + 1

    # posture(2축) 우선 — 없으면 라벨 단일축 fallback (회귀 0)
    _pm = _posture_multiplier(_load_posture())
    regime_mult = _pm if _pm is not None else _regime_multiplier(regime)
    candidates: list[TickerScore] = []
    skipped: list[dict[str, Any]] = []

    for u in active:
        scored, reason = _score_one(
            symbol=u["symbol"],
            tier=u["tier"],
            daily_budget_krw=daily_budget_krw,
            tier_counts=tier_counts,
            regime_mult=regime_mult,
            fetcher=fetcher,
            live_fetcher=live_fetcher,
        )
        if scored is None:
            skipped.append({"symbol": u["symbol"], "reason": reason})
        else:
            candidates.append(scored)

    candidates.sort(key=lambda t: t.score, reverse=True)
    return DCAReport(
        generated_at=datetime.now(KST).isoformat(timespec="seconds"),
        market_window=market_window,
        regime=regime,
        regime_multiplier=regime_mult,
        threshold=SCORE_THRESHOLD,
        daily_budget_krw=daily_budget_krw,
        candidates=candidates,
        skipped=skipped,
    )


# ============================================================
# Discord formatting + push
# ============================================================


def _market_emoji(window: str) -> str:
    return {"KR": "🇰🇷", "US": "🇺🇸", "ALL": "🌐"}.get(window, "")


def _signal_pills(c: TickerScore) -> str:
    parts: list[str] = []
    if c.rsi is not None and c.rsi <= 40:
        parts.append(f"RSI {c.rsi:.0f}")
    if c.ma50_distance_pct is not None and c.ma50_distance_pct <= -5:
        parts.append(f"MA50 {c.ma50_distance_pct:+.1f}%")
    if c.zscore_20d is not None and c.zscore_20d <= -1:
        parts.append(f"z={c.zscore_20d:.1f}σ")
    if c.drawdown_52w_pct is not None and c.drawdown_52w_pct <= -15:
        parts.append(f"52wDD {c.drawdown_52w_pct:.0f}%")
    return " · ".join(parts) or "soft signal"


def _skip_summary(skipped: list[dict[str, Any]]) -> tuple[int, int]:
    """(below_threshold_count, data_error_count) — '점수미달'과 '데이터오류' 구분."""
    data_err = sum(1 for s in skipped if s.get("reason") == "insufficient_history")
    return len(skipped) - data_err, data_err


def format_discord_message(report: DCAReport) -> str:
    win = _market_emoji(report.market_window)
    when = report.generated_at[:16].replace("T", " ")

    if not report.candidates:
        below, data_err = _skip_summary(report.skipped)
        eval_line = f"_Universe 평가 {len(report.skipped)}개 ({below} 점수미달"
        if data_err:
            eval_line += f" / ⚠️ {data_err} 데이터오류"
        eval_line += f") / regime={report.regime or 'N/A'} (×{report.regime_multiplier})_"
        warn = (
            f"\n⚠️ **{data_err}개 종목 데이터 fetch 실패** — 평가 누락. 결과 신뢰도 낮음.\n"
            if data_err else ""
        )
        return (
            f"📉 **DCA 신호 {win} ({report.market_window})** — {when} KST\n\n"
            f"오늘 score≥{report.threshold} 도달 종목 **후보 없음**. "
            f"평소 DCA 보류 또는 현금 유지.\n"
            f"{warn}"
            f"{eval_line}"
        )

    lines = [
        f"📈 **DCA 매수 후보 {win} ({report.market_window})** — {when} KST",
        f"_Regime: {report.regime or 'N/A'} (×{report.regime_multiplier}) · "
        f"일 예산 ₩{report.daily_budget_krw:,} · score≥{report.threshold}_\n",
    ]

    mult_emoji = {2.0: "🔥", 1.5: "⭐", 1.0: "•"}
    tier_label = {1: "T1", 2: "T2", 3: "T3"}

    for c in report.candidates[:10]:
        emoji = mult_emoji.get(c.multiplier, "·")
        close_fmt = f"₩{c.current_price:,.0f}" if c.market == "KR" else f"${c.current_price:,.2f}"
        if c.live_price is not None and c.live_delta_pct is not None:
            live_fmt = f"₩{c.live_price:,.0f}" if c.market == "KR" else f"${c.live_price:,.2f}"
            delta_emoji = "🟢" if c.live_delta_pct > 0 else ("🔴" if c.live_delta_pct < 0 else "⚪")
            price_line = f"종가 {close_fmt} → **실시간 {live_fmt}** {delta_emoji} {c.live_delta_pct:+.2f}%"
        else:
            price_line = f"종가 {close_fmt}"
        lines.append(
            f"{emoji} **{c.symbol}** [{tier_label[c.tier]}] score **{c.score}** "
            f"→ ₩{c.allocation_krw:,} ({c.multiplier:.1f}x)\n"
            f"   {_signal_pills(c)} · {price_line}"
        )

    if len(report.candidates) > 10:
        lines.append(f"\n_+{len(report.candidates) - 10}개 추가 후보 → state/dca_signals.json_")

    lines.append("\n⚠️ Advisory only — 실제 매수는 폐하 판단.")
    return "\n".join(lines)


def _get_webhook_url() -> str | None:
    env = os.environ.get("CORVIN_DISCORD_WEBHOOK")
    if env:
        return env.strip()
    if not CONFIG_FILE.exists():
        return None
    try:
        cfg = json.loads(CONFIG_FILE.read_text())
    except json.JSONDecodeError:
        return None
    url = cfg.get("notification", {}).get("discord_webhook_url")
    return url if isinstance(url, str) and url.startswith("http") else None


def _dca_dedup_key(report: DCAReport) -> str:
    """매일 같은 (date, market_window)에 대해 1회만 push."""
    return f"{report.generated_at[:10]}::{report.market_window}"


def push_dca_report(report: DCAReport) -> bool:
    """Discord push with daily-window dedup. webhook 없으면 file-only."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    key = _dca_dedup_key(report)
    dedup: dict[str, str] = {}
    if DEDUP_FILE.exists():
        try:
            dedup = json.loads(DEDUP_FILE.read_text())
        except json.JSONDecodeError:
            dedup = {}
    if key in dedup:
        log.info("DCA push dedup hit (%s) — skip", key)
        return False

    content = format_discord_message(report)
    delivered = False

    webhook = _get_webhook_url()
    if channels.is_enabled("discord") and webhook:
        parsed = urlparse(webhook)
        if parsed.scheme == "https" and "discord" in parsed.netloc:
            try:
                r = requests.post(webhook, json={"content": content[:1900]}, timeout=15)
                ok = r.status_code in (200, 204)
                delivered = delivered or ok
                if not ok:
                    log.error("DCA push failed: %s %s", r.status_code, r.text[:200])
            except requests.RequestException as e:
                log.error("DCA push exception: %s", e)
        else:
            log.error("invalid Discord webhook URL")

    if channels.is_enabled("imessage"):
        if channels.send_imessage(content):
            delivered = True
            log.info("DCA iMessage 전송 성공 → %s", channels.imessage_recipient())
        else:
            log.warning("DCA iMessage 전송 실패")

    if delivered:
        dedup[key] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        DEDUP_FILE.write_text(json.dumps(dedup, indent=2))

    return delivered


# ============================================================
# Persistence
# ============================================================


def _load_universe() -> list[dict[str, Any]]:
    if not UNIVERSE_FILE.exists():
        log.error("universe.json 없음: %s", UNIVERSE_FILE)
        return []
    try:
        data = json.loads(UNIVERSE_FILE.read_text())
    except json.JSONDecodeError as e:
        log.error("universe.json parse error: %s", e)
        return []
    return [u for u in data.get("tickers", []) if "symbol" in u and "tier" in u]


def _load_daily_budget() -> int:
    if not CONFIG_FILE.exists():
        return DEFAULT_DAILY_BUDGET_KRW
    try:
        cfg = json.loads(CONFIG_FILE.read_text())
    except json.JSONDecodeError:
        return DEFAULT_DAILY_BUDGET_KRW
    return int(cfg.get("dca", {}).get("daily_budget_krw", DEFAULT_DAILY_BUDGET_KRW))


def _load_regime() -> str | None:
    """jarvis regime 라벨 — last_regime.json 또는 latest snapshot에서."""
    last = STATE_DIR / "last_regime.json"
    if last.exists():
        try:
            return json.loads(last.read_text()).get("label")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _persist_report(report: DCAReport) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": report.generated_at,
        "market_window": report.market_window,
        "regime": report.regime,
        "regime_multiplier": report.regime_multiplier,
        "threshold": report.threshold,
        "daily_budget_krw": report.daily_budget_krw,
        "candidates": [asdict(c) for c in report.candidates],
        "skipped": report.skipped,
    }
    SIGNALS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    below, data_err = _skip_summary(report.skipped)
    log.info(
        "dca_signals.json 작성: %d candidates, %d skipped (%d below-threshold, %d data-error)",
        len(report.candidates), len(report.skipped), below, data_err,
    )


# ============================================================
# CLI entry point
# ============================================================


def _live_price_fetcher(symbol: str) -> float | None:
    """quote_provider 위임 — KIS 시 live, 아니면 yfinance EoD가 반환됨."""
    from corvin_jarvis import quote_provider
    q = quote_provider.get_stock_quote(symbol)
    return q.price


def run(market_window: str, *, push: bool = True, prefer_kis: bool = True) -> DCAReport:
    universe = _load_universe()
    if not universe:
        log.error("universe 비어있음 — 종료")
        return DCAReport(
            generated_at=datetime.now(KST).isoformat(timespec="seconds"),
            market_window=market_window,
            regime=None, regime_multiplier=1.0,
            threshold=SCORE_THRESHOLD,
            daily_budget_krw=_load_daily_budget(),
            candidates=[], skipped=[],
        )

    # quote_provider가 KIS 우선 + 폴백을 모두 처리 (single source of truth)
    bar_fetcher: Fetcher = default_fetcher
    live_fetcher: LivePriceFetcher | None = _live_price_fetcher if prefer_kis else None

    report = build_dca_report(
        universe=universe,
        daily_budget_krw=_load_daily_budget(),
        market_window=market_window,
        regime=_load_regime(),
        fetcher=bar_fetcher,
        live_fetcher=live_fetcher,
    )
    _persist_report(report)
    if push:
        push_dca_report(report)
    return report


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Corvin Daily DCA Timing")
    parser.add_argument("--window", choices=["KR", "US", "ALL"], default="ALL")
    parser.add_argument("--dry-run", action="store_true", help="Discord push 생략")
    parser.add_argument(
        "--no-kis", action="store_true",
        help="KIS 비활성화 — yfinance/pykrx만 사용 (디버깅용)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = run(args.window, push=not args.dry_run, prefer_kis=not args.no_kis)
    print("\n" + "=" * 60)
    print(format_discord_message(report))


if __name__ == "__main__":
    _cli()
