"""행동 판정 엔진 — 종목별 매수/홀딩/매도 등 판정 (advisory·모의). 신호 융합."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_STOP_LOSS = -8.0
_TAKE_PROFIT = 25.0
_RS_LAGGARD = -4.0
_RS_LEADER = 4.0
_DCA_DEEP = 75
_DCA_STRONG = 60
_DCA_OK = 50


@dataclass(frozen=True)
class Verdict:
    symbol: str
    action: str       # 매수 | 분할매수 | 홀딩 | 비중축소 | 매도 | 관망
    confidence: str   # 상 | 중 | 하
    rationale: str


def decide(ctx: dict[str, Any]) -> Verdict:
    """신호 컨텍스트를 행동 판정으로 융합 (advisory only)."""
    sym = ctx.get("symbol", "")
    held = bool(ctx.get("held"))
    pnl = ctx.get("pnl_pct")
    dca = int(ctx.get("dca_score") or 0)
    rs = ctx.get("rs")
    today = ctx.get("pct_today")
    alive = bool(ctx.get("theme_alive"))
    hv = bool(ctx.get("high_vol"))
    cap = "중" if hv else "상"   # 무어샷 신뢰도 상한

    def v(action: str, conf: str, why: str) -> Verdict:
        return Verdict(symbol=sym, action=action, confidence=conf, rationale=why)

    if held:
        if pnl is not None and pnl <= _STOP_LOSS:
            return v("매도", "상", f"손절선({_STOP_LOSS}%) 도달 — 리스크 관리 우선")
        if pnl is not None and pnl >= _TAKE_PROFIT:
            return v("비중축소", cap, f"익절선(+{_TAKE_PROFIT}%) 도달 — 일부 차익실현 검토")
        if rs is not None and rs <= _RS_LAGGARD and not alive:
            return v("비중축소", "중", "지수 대비 약세 + 테마 식음 — 비중 점검")
        return v("홀딩", cap if alive else "중",
                 "보유 논리 유효" + (" · 테마 살아있음" if alive else ""))

    # 미보유
    spike = 15.0 if hv else 8.0
    if today is not None and today >= spike:
        return v("관망", "하" if hv else "중", f"오늘 이미 +{today:.0f}% 급등 — 추격 위험, 눌림 대기")
    if dca >= _DCA_DEEP and alive and (rs or 0) > 0:
        return v("매수", cap, f"DCA 가치점수 {dca}(깊은 저평가)+테마 살아있음+주도주")
    if dca >= _DCA_STRONG and alive:
        return v("분할매수", cap if (rs or 0) > 0 else "중",
                 f"DCA 가치점수 {dca}+테마 살아있음 — 분할 진입")
    if dca >= _DCA_OK and (alive or (rs or 0) > 0):
        return v("분할매수", "하" if hv else "중", f"DCA 가치점수 {dca} — 분할 진입 후보")
    if rs is not None and rs >= _RS_LEADER and alive:
        return v("분할매수", "중", "지수 대비 주도주+테마 살아있음(가치점수는 낮음)")
    return v("관망", "하", "뚜렷한 진입 신호 없음 — 관찰")


_SECTOR_TH = {"semiconductor": 3.0, "shipbuilding": 4.0, "defense": 4.0}
_SECTOR_TH_DEFAULT = 4.0


def _dca_value_score(prices: list[float]) -> int:
    """DCA 가치/과매도 점수 (0~100, 높을수록 저평가). 임계 게이트 없이 raw."""
    from corvin_jarvis.dca_timing import (
        MIN_HISTORY_DAYS, _composite_score, _drawdown_52w_pct,
        _ma_distance_pct, _rsi, _zscore,
    )
    if len(prices) < MIN_HISTORY_DAYS:
        return 0
    score, _ = _composite_score(
        _rsi(prices), _ma_distance_pct(prices, 50),
        _zscore(prices, 20), _drawdown_52w_pct(prices),
    )
    return int(score)


def _sector_of(latest: dict[str, Any], symbol: str) -> str | None:
    for u in latest.get("universe", []):
        if u.get("symbol") == symbol:
            return u.get("sector")
    return None


def _theme_alive(latest: dict[str, Any], sector: str | None) -> bool:
    if not sector:
        return False
    pcts = [u.get("pct_change") for u in latest.get("universe", [])
            if u.get("sector") == sector and u.get("pct_change") is not None]
    if len(pcts) < 2:
        return False
    avg = sum(pcts) / len(pcts)
    return avg >= _SECTOR_TH.get(sector, _SECTOR_TH_DEFAULT)


def for_symbol(symbol: str, latest: dict[str, Any]) -> Verdict:
    """종목 + 최신 snapshot으로 컨텍스트 조립 후 판정."""
    from corvin_jarvis import dca_timing, quote_provider

    q = quote_provider.get_stock_quote(symbol)
    pct_today = q.pct_change
    dca_score = _dca_value_score(dca_timing.default_fetcher(symbol))

    is_kr = dca_timing._is_kr_symbol(symbol)
    idx_name = "kospi" if is_kr else "sp500"
    idx_pct = (latest.get("indices", {}).get(idx_name, {}) or {}).get("pct_change")
    rs = (pct_today - idx_pct) if (pct_today is not None and idx_pct is not None) else None

    held, pnl = False, None
    for p in latest.get("portfolio", []):
        if p.get("symbol") == symbol:
            held, pnl = True, p.get("pnl_pct")
            break

    sector = _sector_of(latest, symbol)
    ctx = {
        "symbol": symbol, "held": held, "pnl_pct": pnl, "dca_score": dca_score,
        "rs": rs, "pct_today": pct_today, "theme_alive": _theme_alive(latest, sector),
        "high_vol": sector is None,   # monitored_universe 미포함 = 무어샷(보수)
    }
    return decide(ctx)


def verdicts_for_state(latest: dict[str, Any], alerts: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """보유 종목 + 오늘 alert이 가리키는 개별 종목에 대해 verdict 일괄 산출."""
    syms: set[str] = set()
    for p in latest.get("portfolio", []):
        if p.get("symbol"):
            syms.add(str(p["symbol"]))
    for a in alerts:
        cat = a.get("category", "")
        parts = a.get("metric", "").split("_")
        if cat in ("universe", "leading_rs") and len(parts) >= 2:
            syms.add(parts[1])
        elif cat == "portfolio" and parts:
            syms.add(parts[-1])

    out: dict[str, dict[str, str]] = {}
    for s in sorted(syms):
        v = for_symbol(s, latest)
        out[s] = {"action": v.action, "confidence": v.confidence, "rationale": v.rationale}
    return out
