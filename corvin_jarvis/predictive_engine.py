"""STAGE①.5 예측 시그널 엔진 — Reactive에서 Predictive로 확장.

세 Pillar 통합:
  VELOCITY — 가격 하락 속도(선형 기울기) → 손절선 도달 예상일
  RS_WEAK  — 보유종목 n일 수익률 - 벤치마크 수익률 < threshold → 상대강도 약화
  EVENT    — FOMC/BOK 등 거시 일정 D-N 선제 경보 (방향성 없음, 변동성 준비)

순수 함수 · 부작용 없음 · 실주문 0. closes_by_sym 주입으로 테스트 가능.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from corvin_jarvis.signals.event_calendar import macro_events_within

_VELOCITY_HORIZON = 14     # 이 일수 이내 도달 예상이면 경보
_NOISE_GATE = 0.15         # |기울기|/ATR프록시 미만 = 변동성 노이즈 → 억제
_ATR_WINDOW = 14


def _fmt(v: float) -> str:
    """가격 가독 포맷 — KRW 큰 수는 천단위, USD는 소수 유지. 1e+06 방지."""
    return f"{round(v):,}" if v >= 10000 else f"{v:g}"


_RS_THRESHOLD = -5.0       # %p 이하면 상대강도 약세
_RS_N_DAYS = 20
_RS_HIST_WINDOWS = 60      # 적응 임계 계산용 과거 rs 표본 수
_RS_MIN_SAMPLES = 30       # 미만이면 기본 임계 fallback
_RS_CLAMP = (-20.0, -2.0)  # 적응 임계 안전 클램프
_EVENT_HORIZON = 14

_DEFAULT_CONFIDENCE = {"VELOCITY": 65.0, "RS_WEAK": 60.0, "EVENT": 90.0}
_CALIBRATION_PATH = Path(__file__).resolve().parent / "state" / "calibration.json"


@dataclass(frozen=True)
class PredictiveSignal:
    symbol: str            # "" = 매크로 이벤트 (종목 무관)
    kind: str              # VELOCITY | RS_WEAK | EVENT
    urgency: int           # 0-100
    confidence: float      # 0-100
    horizon_days: int | None
    message: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── 공통 헬퍼 ─────────────────────────────────────────────

def _slope(closes: list[float]) -> float | None:
    """oldest→newest 종가 리스트의 1일 평균 변화량(가격 단위)."""
    if len(closes) < 3:
        return None
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    return statistics.mean(changes)


def _atr_proxy(closes: list[float], window: int = _ATR_WINDOW) -> float | None:
    """종가 기반 변동성 프록시 — 최근 window일 |일변화| 평균.

    데이터 제약: quote_provider가 종가만 제공(OHLC 없음) → 진짜 ATR 대신
    close-to-close 변동성으로 대체 (스펙 §6 의도 동일: 변동성 정규화).
    window+1봉 미만이거나 변동 0이면 None.
    """
    if len(closes) < window + 1:
        return None
    tail = closes[-(window + 1):]
    changes = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
    atr = statistics.mean(changes)
    return atr if atr > 0 else None


def _slope_se(closes: list[float]) -> float:
    """일변화량 평균의 표준오차 — 신뢰구간용. 변화 표본<2 → 0."""
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    if len(changes) < 2:
        return 0.0
    return statistics.stdev(changes) / math.sqrt(len(changes))


def _n_day_return(closes: list[float], n: int) -> float | None:
    """최근 n봉 수익률(%). 데이터 부족 시 None."""
    if len(closes) < n + 1:
        return None
    old, new = closes[-(n + 1)], closes[-1]
    if old <= 0:
        return None
    return (new / old - 1) * 100.0


def _load_calibration(path: Path | None = None) -> dict:
    """Phase 1 scorer cron이 쓰는 state/calibration.json 로드. 없으면 {}."""
    try:
        return json.loads(Path(path or _CALIBRATION_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def confidence_for(kind: str, calibration: dict | None = None,
                   engine: str = "predictive") -> float:
    """적중률 보정 confidence — calibrated 없으면(n<10 포함) 기본값 fallback."""
    cal = calibration if calibration is not None else _load_calibration()
    entry = (cal.get(engine) or {}).get(kind) or {}
    c = entry.get("calibrated_confidence")
    return float(c) if c is not None else _DEFAULT_CONFIDENCE.get(kind, 50.0)


# ── VELOCITY ──────────────────────────────────────────────

def evaluate_velocity(
    holdings: list[dict[str, Any]],
    stops: dict[str, float],
    closes_by_sym: dict[str, list[float]],
    horizon: int = _VELOCITY_HORIZON,
    confidence: float | None = None,
) -> list[PredictiveSignal]:
    """하락 추세 기울기로 손절선 도달 예상일 경보 (Phase 3: 노이즈 게이트 + 신뢰구간).

    이미 손절선 이하인 경우는 signal_engine(STOP)이 담당 — 여기선 불개입.
    """
    out: list[PredictiveSignal] = []
    conf = confidence if confidence is not None else _DEFAULT_CONFIDENCE["VELOCITY"]
    for pos in holdings:
        sym = str(pos.get("symbol", ""))
        stop = stops.get(sym)
        if stop is None:
            continue
        closes = closes_by_sym.get(sym, [])
        s = _slope(closes)
        if s is None or s >= 0:
            continue  # 상승·횡보 추세
        price = pos.get("price") or (closes[-1] if closes else None)
        if price is None or price <= stop:
            continue  # 이미 손절 이탈 → signal_engine 담당
        atr = _atr_proxy(closes)
        strength = (-s / atr) if atr else None
        if strength is not None and strength < _NOISE_GATE:
            continue  # 변동성 대비 미미한 기울기 — 노이즈 억제
        dist = price - stop
        days_to = dist / (-s)
        if days_to > horizon:
            continue
        se = _slope_se(closes)
        days_lo = round(dist / (-s + se), 1) if (-s + se) > 0 else None  # 빠른 시나리오
        days_hi = round(dist / (-s - se), 1) if (-s - se) > 0 else None  # 느린 시나리오
        rng = ""
        if days_lo is not None and days_hi is not None and days_lo != days_hi:
            rng = f" (범위 {math.floor(days_lo)}–{math.ceil(days_hi)}일)"
        urgency = max(40, min(85, int(85 - (days_to / horizon) * 45)))
        out.append(PredictiveSignal(
            symbol=sym, kind="VELOCITY", urgency=urgency, confidence=conf,
            horizon_days=round(days_to),
            message=f"하락 속도 기준 손절선({_fmt(stop)}) ~{round(days_to)}일 내 도달 예상{rng}",
            evidence={"slope_per_day": round(s, 4), "days_to_stop": round(days_to, 1),
                      "stop": stop, "current_price": price,
                      "atr_proxy": round(atr, 4) if atr else None,
                      "strength": round(strength, 3) if strength is not None else None,
                      "days_lo": days_lo, "days_hi": days_hi},
        ))
    return out


# ── RS_WEAK ────────────────────────────────────────────────

def _adaptive_rs_threshold(closes: list[float], bench: list[float],
                           n_days: int = _RS_N_DAYS,
                           default: float = _RS_THRESHOLD) -> float:
    """종목별 과거 rs 분포의 하위 10분위 — 변동성 맞춤 임계 (스펙 §6).

    rs_i = (종목 n일 수익률 - 벤치 n일 수익률), 과거 _RS_HIST_WINDOWS개 윈도.
    표본 < _RS_MIN_SAMPLES → default(-5.0). 결과는 _RS_CLAMP로 클램프.
    """
    rs_vals: list[float] = []
    for i in range(_RS_HIST_WINDOWS):
        end_c, end_b = len(closes) - i, len(bench) - i
        h = _n_day_return(closes[:end_c], n_days)
        b = _n_day_return(bench[:end_b], n_days)
        if h is None or b is None:
            break
        rs_vals.append(h - b)
    if len(rs_vals) < _RS_MIN_SAMPLES:
        return default
    rs_sorted = sorted(rs_vals)
    thr = rs_sorted[int(len(rs_sorted) * 0.10)]
    lo, hi = _RS_CLAMP
    return max(lo, min(hi, thr))


def evaluate_relative_strength(
    holdings: list[dict[str, Any]],
    closes_by_sym: dict[str, list[float]],
    bench_closes_by_market: dict[str, list[float]],
    n_days: int = _RS_N_DAYS,
    threshold: float | None = None,
    confidence: float | None = None,
) -> list[PredictiveSignal]:
    """보유종목 n일 수익률 - 벤치마크 수익률 < threshold%p → RS_WEAK.

    threshold=None이면 종목별 적응 임계(_adaptive_rs_threshold) 사용 —
    이력 부족 시 기본 _RS_THRESHOLD(-5.0)로 fallback해 기존 동작 보존.
    """
    out: list[PredictiveSignal] = []
    conf = confidence if confidence is not None else _DEFAULT_CONFIDENCE["RS_WEAK"]
    for pos in holdings:
        sym = str(pos.get("symbol", ""))
        market = str(pos.get("market", "US"))
        closes = closes_by_sym.get(sym, [])
        bench = bench_closes_by_market.get(market, [])
        h_ret = _n_day_return(closes, n_days)
        b_ret = _n_day_return(bench, n_days)
        if h_ret is None or b_ret is None:
            continue
        rs = h_ret - b_ret
        thr = threshold if threshold is not None else _adaptive_rs_threshold(closes, bench)
        if rs >= thr:
            continue
        urgency = max(30, min(75, int(30 + (thr - rs) * 4)))
        bench_label = "KOSPI" if market == "KR" else "S&P500"
        out.append(PredictiveSignal(
            symbol=sym, kind="RS_WEAK", urgency=urgency, confidence=conf,
            horizon_days=None,
            message=f"{n_days}일 {bench_label} 대비 상대강도 {rs:+.1f}%p — 약세 심화 추세",
            evidence={"holding_ret_pct": round(h_ret, 2), "bench_ret_pct": round(b_ret, 2),
                      "rs_pct": round(rs, 2), "n_days": n_days,
                      "threshold_pct": round(thr, 2)},
        ))
    return out


# ── EVENT ──────────────────────────────────────────────────

def evaluate_events(
    as_of: date,
    held_symbols: list[str],
    horizon_days: int = _EVENT_HORIZON,
    confidence: float | None = None,
) -> list[PredictiveSignal]:
    """FOMC/BOK D-N 선제 경보. 방향성 없음 — 변동성 준비 신호."""
    conf = confidence if confidence is not None else _DEFAULT_CONFIDENCE["EVENT"]
    events = macro_events_within(as_of, horizon_days=horizon_days)
    out: list[PredictiveSignal] = []
    for ev in events:
        d = ev["days_to"]
        urgency = 80 if d <= 1 else (65 if d <= 3 else 50)
        out.append(PredictiveSignal(
            symbol="", kind="EVENT", urgency=urgency, confidence=conf,
            horizon_days=d,
            message=f"{ev['name']} D-{d} — 장중 변동성 확대 가능",
            evidence={"event": ev["name"], "event_date": str(ev["event_date"]), "days_to": d},
        ))
    return sorted(out, key=lambda s: s.horizon_days or 99)


# ── 통합 ──────────────────────────────────────────────────

def evaluate(
    holdings: list[dict[str, Any]],
    *,
    stops: dict[str, float] | None = None,
    closes_by_sym: dict[str, list[float]] | None = None,
    bench_closes_by_market: dict[str, list[float]] | None = None,
    as_of: date | None = None,
    calibration: dict | None = None,
) -> list[PredictiveSignal]:
    """세 Pillar 통합 → 긴급도 내림차순. confidence는 적중률 보정값 주입."""
    from corvin_jarvis.signal_engine import load_stops
    _stops = stops if stops is not None else load_stops()
    _closes = closes_by_sym or {}
    _bench = bench_closes_by_market or {}
    _date = as_of or date.today()
    _cal = calibration if calibration is not None else _load_calibration()

    sigs: list[PredictiveSignal] = []
    sigs.extend(evaluate_velocity(holdings, _stops, _closes,
                                  confidence=confidence_for("VELOCITY", _cal)))
    sigs.extend(evaluate_relative_strength(holdings, _closes, _bench,
                                           confidence=confidence_for("RS_WEAK", _cal)))
    sigs.extend(evaluate_events(_date, [str(p.get("symbol", "")) for p in holdings],
                                confidence=confidence_for("EVENT", _cal)))
    return sorted(sigs, key=lambda s: s.urgency, reverse=True)
