"""행동 판정 엔진 — 종목별 매수/홀딩/매도 등 판정 (advisory·모의). 신호 융합."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_STOP_LOSS = -8.0          # ATR 미가용 시 평면 fallback (A 버킷)
_ATR_STOP_K = 5.0          # 손절선 = -(K × 일일 ATR%) — 변동성 조정
# 음수 손절 공간: -25=가장 깊은(넓은) 손절, -4=가장 얕은(타이트한) 손절.
# (widest, tightest) 순서. 튜플 순서 바꾸면 클램프가 역전되니 주의.
_ATR_STOP_CLAMP = (-25.0, -4.0)
_TAKE_PROFIT = 25.0        # peak 미상 시 평면 fallback
_TP_ACTIVATE = 15.0        # 이 수익% 넘어야 추적 익절 가동 (그 전엔 승자 달리게)
_TP_GIVEBACK_K = 3.0       # 고점에서 K×ATR% 되돌리면 익절 (이익 잠금)
_TP_MIN_GIVEBACK = 5.0     # ATR 없을 때 최소 되돌림%
_TP_PEAK_WINDOW = 15       # peak는 최근 N봉 스윙 고점만 (진입 전 고점 오염 방지)
_TP_BIG = 40.0             # 고점 수익% 이상이면 익절 시 절반(아니면 1/3) — 분할 사다리
_ADD_SPIKE_CAP = 8.0       # 오늘 이 % 이상 급등이면 불타기 보류 (FOMO 추격 방지)
_RS_LAGGARD = -4.0
_RS_LEADER = 4.0
_DCA_DEEP = 75
_DCA_STRONG = 60
_DCA_OK = 50


@dataclass(frozen=True)
class Verdict:
    symbol: str
    action: str       # 매수 | 분할매수 | 비중확대 | 홀딩 | 비중축소 | 매도 | 관망
    confidence: str   # 상 | 중 | 하
    rationale: str


def _atr_stop_threshold(atr_pct: float | None,
                        k: float = _ATR_STOP_K) -> float | None:
    """변동성 조정 손절 임계(%). atr_pct = 일일 ATR / 가격 × 100.

    None/0 이하면 None → 호출측이 평면 _STOP_LOSS로 fallback.
    결과는 _ATR_STOP_CLAMP로 클램프 (고변동=넓게, 저변동=타이트하게).
    """
    if not atr_pct or atr_pct <= 0:
        return None
    widest, tightest = _ATR_STOP_CLAMP   # (-25, -4): 음수 공간 floor/ceiling
    return max(widest, min(tightest, -k * atr_pct))


def _peak_pnl_pct(closes: list[float], price: float | None,
                  pnl_pct: float | None, window: int = _TP_PEAK_WINDOW) -> float | None:
    """최근 window 봉 스윙 고점 기준 peak PnL% — 추적 익절 기준값(무상태).

    avg는 price/pnl에서 유도(진입가 필드 불요). *최근* 고점만 써서 진입 전
    고점으로 인한 거짓 되돌림(조기 익절)을 방지 — 추적 손절의 정석 의미.
    """
    if pnl_pct is None or pnl_pct <= -100 or price is None or not closes:
        return None
    avg_implied = price / (1 + pnl_pct / 100)
    if avg_implied <= 0:
        return None
    peak_close = max(closes[-window:] + [price])
    return (peak_close / avg_implied - 1) * 100


def _trailing_take_profit(pnl_pct: float | None, peak_pnl_pct: float | None,
                          atr_pct: float | None, activate: float = _TP_ACTIVATE,
                          k: float = _TP_GIVEBACK_K) -> bool:
    """추적 익절 판정 — 고점 대비 되돌림이 ATR 비례 임계 넘으면 True.

    peak None → 평면 _TAKE_PROFIT fallback(기존 동작 보존).
    peak < activate → False (수익 미미, 승자 달리게).
    되돌림(peak−pnl) ≥ max(_TP_MIN_GIVEBACK, k×atr_pct) → 익절(이익 잠금).
    """
    if pnl_pct is None:
        return False
    if peak_pnl_pct is None:
        return pnl_pct >= _TAKE_PROFIT
    peak = max(peak_pnl_pct, pnl_pct)
    if peak < activate:
        return False
    trail = max(_TP_MIN_GIVEBACK, k * atr_pct) if atr_pct else _TP_MIN_GIVEBACK
    return (peak - pnl_pct) >= trail


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
        # 버킷: A(trade, 기본)=변동성 조정 가격 손절 / B(dca)=가격 손절 없음(thesis 기준)
        bucket = str(ctx.get("bucket") or "trade")
        if bucket != "dca" and pnl is not None:
            atr_stop = _atr_stop_threshold(ctx.get("atr_pct"))
            stop_thr = atr_stop if atr_stop is not None else _STOP_LOSS
            if pnl <= stop_thr:
                why = (f"ATR 손절({stop_thr:.1f}%, {_ATR_STOP_K:g}×ATR) 도달 — 리스크 관리"
                       if atr_stop is not None
                       else f"손절선({_STOP_LOSS}%) 도달 — 리스크 관리 우선")
                return v("매도", "상", why)
        peak = ctx.get("peak_pnl_pct")
        if _trailing_take_profit(pnl, peak, ctx.get("atr_pct")):
            if peak is not None:
                eff_peak = max(peak, pnl)
                tranche = "절반" if eff_peak >= _TP_BIG else "1/3"   # ④ 분할 사다리
                why = (f"추적 익절 — 고점 +{eff_peak:.0f}%에서 되돌림, "
                       f"{tranche} 차익실현(잔여 추적 유지)")
            else:
                why = f"익절선(+{_TAKE_PROFIT}%) 도달 — 일부 차익실현 검토"
            return v("비중축소", cap, why)
        if rs is not None and rs <= _RS_LAGGARD and not alive:
            suffix = " — DCA(B) thesis 점검" if bucket == "dca" else ""
            return v("비중축소", "중", "지수 대비 약세 + 테마 식음 — 비중 점검" + suffix)
        # ③ 불타기 — 강세 지속 승자에 분할 추가 (trade 전용: 모멘텀 vs DCA 가치 분리)
        if (bucket != "dca" and pnl is not None and pnl > 0 and alive
                and rs is not None and rs >= _RS_LEADER
                and (today is None or today < _ADD_SPIKE_CAP)):
            return v("비중확대", cap,
                     "강세 지속(지수 주도+테마 살아있음) — 불타기 후보(분할 추가)")
        if bucket == "dca":
            return v("홀딩", cap if alive else "중",
                     "DCA(B) 보유 · 가격손절 없음(예약추매/thesis 기준)"
                     + (" · 테마 살아있음" if alive else ""))
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


_SECTOR_TH = {"semiconductor": 3.0, "shipbuilding": 4.0, "defense": 4.0,
              "humanoid": 8.0, "space": 8.0, "stem_cell": 8.0, "quantum": 10.0}
_SECTOR_TH_DEFAULT = 4.0
_MOONSHOT_SECTORS = {"humanoid", "space", "stem_cell", "quantum"}


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
    closes = dca_timing.default_fetcher(symbol)
    dca_score = _dca_value_score(closes)

    is_kr = dca_timing._is_kr_symbol(symbol)
    idx_name = "kospi" if is_kr else "sp500"
    idx_pct = (latest.get("indices", {}).get(idx_name, {}) or {}).get("pct_change")
    rs = (pct_today - idx_pct) if (pct_today is not None and idx_pct is not None) else None

    held, pnl, bucket = False, None, "trade"
    for p in latest.get("portfolio", []):
        if p.get("symbol") == symbol:
            held, pnl = True, p.get("pnl_pct")
            bucket = str(p.get("bucket") or "trade")  # 기본 trade(A)=보호적 손절
            break

    # ATR%(일일) + peak PnL — 손절(A버킷)·추적익절(전버킷) 공용, 가져온 closes 재사용
    atr_pct = peak_pnl_pct = None
    if held and closes and q.price:
        from corvin_jarvis import predictive_engine as _pe
        atr = _pe._atr_proxy(closes)
        if atr:
            atr_pct = atr / q.price * 100
        peak_pnl_pct = _peak_pnl_pct(closes, q.price, pnl)  # 최근 스윙 고점 기준

    sector = _sector_of(latest, symbol)
    ctx = {
        "symbol": symbol, "held": held, "pnl_pct": pnl, "dca_score": dca_score,
        "rs": rs, "pct_today": pct_today, "theme_alive": _theme_alive(latest, sector),
        "high_vol": sector is None or sector in _MOONSHOT_SECTORS,  # 미추적 or 미래기술 무어샷 = 보수
        "bucket": bucket, "atr_pct": atr_pct, "peak_pnl_pct": peak_pnl_pct,
    }
    return decide(ctx)


def verdicts_for_state(latest: dict[str, Any], alerts: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """보유 종목 + 오늘 alert이 가리키는 개별 종목에 대해 verdict 일괄 산출 (회사명 포함)."""
    from corvin_jarvis.signals import universe_loader
    name_map = {t.symbol: t.name for t in universe_loader.load()}

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
        out[s] = {"action": v.action, "confidence": v.confidence,
                  "rationale": v.rationale, "name": name_map.get(s, "")}
    return out
