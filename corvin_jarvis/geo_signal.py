"""Corvin Jarvis — Geopolitical / Risk Signal (Phase 7)

외부 MCP 의존 없이 composite risk score 계산.
사용 데이터: VIX, Gold/Oil ratio, DXY, BTC (모두 pulse가 이미 수집).
추가 fetch: BTC, S&P sector ETFs (defensive vs offensive).

산출:
    state/geo_signal.json — risk_score, regime, detail

사용:
    python corvin_jarvis/geo_signal.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from corvin_jarvis import quote_provider

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
LATEST_FILE = STATE_DIR / "latest.json"
SIGNAL_FILE = STATE_DIR / "geo_signal.json"
LOG_FILE = STATE_DIR / "geo_signal.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("corvin.geo")


@dataclass(frozen=True)
class RiskComponent:
    name: str
    value: float | None
    score: float  # 0~10, 높을수록 risk-off
    note: str


@dataclass
class RiskSignal:
    timestamp: str
    composite_score: float
    regime: str
    components: list[dict[str, Any]]
    headline_drivers: list[str]


def _fetch_btc() -> dict[str, Any]:
    q = quote_provider.get_crypto_quote("BTC-USD")
    if q.error or q.price is None:
        return {"price": None, "pct_change": None, "error": q.error or "no data"}
    return {"price": round(q.price, 2), "pct_change": round(q.pct_change or 0.0, 2)}


def _fetch_defensive_sectors() -> dict[str, Any]:
    """방어 vs 공격 sector ETF — XLU(유틸)/XLY(소비), XLP(생활)/XLK(테크).

    ETF는 quote_provider 통해 KIS overseas → yfinance 폴백.
    """
    out: dict[str, Any] = {}
    pairs = {"xlu_xly": ("XLU", "XLY"), "xlp_xlk": ("XLP", "XLK")}
    for name, (def_t, off_t) in pairs.items():
        d_q = quote_provider.get_stock_quote(def_t)
        o_q = quote_provider.get_stock_quote(off_t)
        if d_q.price is None or o_q.price is None:
            out[name] = {"error": d_q.error or o_q.error or "no data"}
            continue
        d_pct = d_q.pct_change or 0.0
        o_pct = o_q.pct_change or 0.0
        out[name] = {
            "defensive_pct": round(d_pct, 2),
            "offensive_pct": round(o_pct, 2),
            "rotation": round(d_pct - o_pct, 2),
        }
    return out


def _score_vix(vix: float | None) -> RiskComponent:
    if vix is None:
        return RiskComponent("vix", None, 5, "데이터 없음 — neutral")
    if vix >= 30:
        return RiskComponent("vix", vix, 10, f"VIX {vix:.1f} — 공포 극단")
    if vix >= 25:
        return RiskComponent("vix", vix, 8, f"VIX {vix:.1f} — 공포")
    if vix >= 20:
        return RiskComponent("vix", vix, 6, f"VIX {vix:.1f} — 경계")
    if vix >= 15:
        return RiskComponent("vix", vix, 4, f"VIX {vix:.1f} — 보통")
    return RiskComponent("vix", vix, 2, f"VIX {vix:.1f} — 안정")


def _score_dxy(dxy: float | None, pct: float | None) -> RiskComponent:
    if dxy is None:
        return RiskComponent("dxy", None, 5, "데이터 없음")
    if dxy >= 105:
        return RiskComponent("dxy", dxy, 7, f"DXY {dxy:.1f} — 강달러 → EM/한국 자본유출 압력")
    if dxy >= 100:
        return RiskComponent("dxy", dxy, 5, f"DXY {dxy:.1f} — 중립")
    return RiskComponent("dxy", dxy, 3, f"DXY {dxy:.1f} — 약달러 → risk-on 우호")


def _score_oil(oil_pct: float | None) -> RiskComponent:
    if oil_pct is None:
        return RiskComponent("oil_momentum", None, 5, "데이터 없음")
    if oil_pct >= 5:
        return RiskComponent("oil_momentum", oil_pct, 8, f"Brent {oil_pct:+.1f}% — 지정학 프리미엄 / 인플레 압력")
    if oil_pct >= 2:
        return RiskComponent("oil_momentum", oil_pct, 6, f"Brent {oil_pct:+.1f}% — 상방 압력")
    if oil_pct <= -5:
        return RiskComponent("oil_momentum", oil_pct, 7, f"Brent {oil_pct:+.1f}% — 수요 둔화 우려")
    return RiskComponent("oil_momentum", oil_pct, 4, f"Brent {oil_pct:+.1f}% — 정상 범위")


def _score_gold_copper(gold_pct: float | None, copper_pct: float | None) -> RiskComponent:
    """금 ↑ + 구리 ↓ = risk-off 시그널 (전통적 macro 로직)."""
    if gold_pct is None or copper_pct is None:
        return RiskComponent("gold_copper_ratio", None, 5, "데이터 없음")
    spread = gold_pct - copper_pct
    if spread >= 3:
        return RiskComponent("gold_copper_ratio", spread, 8, f"Gold-Copper spread {spread:+.1f}% — 강한 risk-off")
    if spread >= 1:
        return RiskComponent("gold_copper_ratio", spread, 6, f"spread {spread:+.1f}% — risk-off 기울임")
    if spread <= -3:
        return RiskComponent("gold_copper_ratio", spread, 3, f"spread {spread:+.1f}% — risk-on 신호")
    return RiskComponent("gold_copper_ratio", spread, 5, f"spread {spread:+.1f}% — 중립")


def _score_btc(btc_pct: float | None) -> RiskComponent:
    if btc_pct is None:
        return RiskComponent("btc_momentum", None, 5, "데이터 없음")
    if btc_pct <= -5:
        return RiskComponent("btc_momentum", btc_pct, 7, f"BTC {btc_pct:+.1f}% — risk-off 동조")
    if btc_pct >= 5:
        return RiskComponent("btc_momentum", btc_pct, 3, f"BTC {btc_pct:+.1f}% — risk-on")
    return RiskComponent("btc_momentum", btc_pct, 5, f"BTC {btc_pct:+.1f}% — 중립")


def _score_rotation(rotations: dict[str, Any]) -> RiskComponent:
    """방어주 - 공격주 rotation. 양수 = defensive 우위 = risk-off."""
    vals = [v["rotation"] for v in rotations.values() if isinstance(v, dict) and "rotation" in v]
    if not vals:
        return RiskComponent("sector_rotation", None, 5, "데이터 없음")
    avg = sum(vals) / len(vals)
    if avg >= 1:
        return RiskComponent("sector_rotation", avg, 7, f"Defensive-Offensive {avg:+.2f}% — 방어주 회피")
    if avg <= -1:
        return RiskComponent("sector_rotation", avg, 3, f"Defensive-Offensive {avg:+.2f}% — 공격주 매수")
    return RiskComponent("sector_rotation", avg, 5, f"rotation {avg:+.2f}% — 중립")


def _regime_from_score(score: float) -> str:
    if score >= 7.5:
        return "CRISIS"
    if score >= 6.0:
        return "RISK_OFF"
    if score >= 4.5:
        return "NEUTRAL"
    if score >= 3.0:
        return "RISK_ON"
    return "EUPHORIA"


def build_geo_signal() -> RiskSignal:
    if not LATEST_FILE.exists():
        log.error("latest.json 없음 — pulse 먼저 실행")
        sys.exit(1)
    latest = json.loads(LATEST_FILE.read_text())

    vix = latest.get("indices", {}).get("vix", {}).get("price")
    dxy = latest.get("fx", {}).get("dxy", {}).get("price")
    dxy_pct = latest.get("fx", {}).get("dxy", {}).get("pct_change")
    oil_pct = latest.get("commodities", {}).get("brent", {}).get("pct_change")
    gold_pct = latest.get("commodities", {}).get("gold", {}).get("pct_change")
    copper_pct = latest.get("commodities", {}).get("copper", {}).get("pct_change")

    btc = _fetch_btc()
    rotations = _fetch_defensive_sectors()

    components = [
        _score_vix(vix),
        _score_dxy(dxy, dxy_pct),
        _score_oil(oil_pct),
        _score_gold_copper(gold_pct, copper_pct),
        _score_btc(btc.get("pct_change")),
        _score_rotation(rotations),
    ]

    composite = sum(c.score for c in components) / len(components)
    regime = _regime_from_score(composite)

    drivers = [c.note for c in components if c.score >= 7]
    if not drivers:
        drivers = [c.note for c in sorted(components, key=lambda x: -x.score)[:2]]

    signal = RiskSignal(
        timestamp=datetime.now().isoformat(),
        composite_score=round(composite, 2),
        regime=regime,
        components=[asdict(c) for c in components],
        headline_drivers=drivers,
    )

    SIGNAL_FILE.write_text(json.dumps(asdict(signal), indent=2, ensure_ascii=False))
    log.info("Composite risk score: %.2f / 10 — regime=%s", composite, regime)
    for c in components:
        log.info("  - %s: %.1f (%s)", c.name, c.score, c.note)
    return signal


if __name__ == "__main__":
    build_geo_signal()
