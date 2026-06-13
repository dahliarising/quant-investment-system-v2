# corvin_jarvis/prediction/run_prediction_digest.py
"""오케스트레이터 — backfill → 5모듈 → 조립 → 전송. launchd 09:36 KST 호출.

--dry-run: 전송 대신 stdout. 모듈 예외는 격리(한 모듈 실패해도 나머지 진행).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from corvin_jarvis.prediction import (backfill, backtest, digest_assembler,
                                      m_band, m_ensemble, m_geopolitical,
                                      m_logistic, m_momentum, m_montecarlo,
                                      m_probability, m_sentiment, m_velocity,
                                      m_vector)
from corvin_jarvis.prediction.contract import PredictionResult

KST = ZoneInfo("Asia/Seoul")
_DB = Path(__file__).resolve().parent.parent / "state" / "daily_history.db"
_FEATURES = ["kospi", "nasdaq", "vix", "usd_krw", "gold", "copper", "dxy"]


def _safe(label: str, fn, *args, **kwargs) -> list[PredictionResult]:
    try:
        out = fn(*args, **kwargs)
        return out if isinstance(out, list) else [out]
    except Exception as e:  # 모듈 격리
        return [PredictionResult(label, "market", f"모듈 오류·생략 ({e})", 0.0,
                                 {}, data_ok=False)]


def _run_velocity(holdings, stops, closes_by_sym):
    return m_velocity.run(holdings, stops, closes_by_sym)


def build_results(*, holdings, universe, geo_payload, stops, closes_by_sym,
                  daily_by_feature, sentiment_payload=None, gate=None,
                  respect_gate: bool = True):
    """5+5 시스템을 실행해 PredictionResult 리스트 + 보유심볼 반환.

    respect_gate=True(다이제스트): 백테스트 통과 모델만 산출.
    respect_gate=False(모델별 뷰): 게이트 무관 전부 산출(탈락도 raw 표시용).
    """
    gate = gate if gate is not None else backtest.load_gate()

    def gated(name: str) -> bool:
        return (not respect_gate) or backtest.passed(gate, name)

    results: list[PredictionResult] = []
    # ── 보유 신호 ──
    results += _safe("velocity", _run_velocity, holdings, stops, closes_by_sym)
    holding_syms = [str(pos.get("symbol", "")) for pos in holdings if pos.get("symbol")]
    for sym in holding_syms:
        results += _safe("probability", m_probability.run_symbol, sym,
                         list(closes_by_sym.get(sym, [])), stops.get(sym, 0.0))
    if gated("montecarlo"):
        for sym in holding_syms:
            results += _safe("montecarlo", m_montecarlo.run_symbol, sym,
                             list(closes_by_sym.get(sym, [])), stops.get(sym))
    # momentum = 보유 우선 → 유니버스 (dedup)
    mom_syms = list(dict.fromkeys(holding_syms + list(universe)))
    for sym in mom_syms:
        results += _safe("momentum", m_momentum.run_symbol, sym,
                         closes_by_sym.get(sym, []))
    # ── 시장 방향 신호 ──
    results += _safe("geopolitical", m_geopolitical.run, geo_payload)
    results += _safe("sentiment", m_sentiment.run, sentiment_payload)
    results += _safe("vector_analog", m_vector.predict, daily_by_feature,
                     features=_FEATURES)
    if gated("logistic"):
        results += _safe("logistic", m_logistic.predict_market, daily_by_feature,
                         features=_FEATURES)
    if not respect_gate:  # 모델별 뷰에선 band도 시장레벨로 산출
        base_closes = [d["close"] for d in daily_by_feature.get(_FEATURES[0], [])]
        results += _safe("band", m_band.run_symbol, "market", base_closes)
    # ── 앙상블 합의 (시장 방향 종합, 게이트 존중) ──
    ens = m_ensemble.run([r for r in results if r.scope == "market"], gate=gate)
    final = ([ens] if ens.data_ok else []) + results
    return final, holding_syms


def build_digest(*, date_str, holdings, universe, db_path, geo_payload,
                 stops, closes_by_sym, daily_by_feature,
                 sentiment_payload=None, gate=None) -> str:
    final, holding_syms = build_results(
        holdings=holdings, universe=universe, geo_payload=geo_payload,
        stops=stops, closes_by_sym=closes_by_sym,
        daily_by_feature=daily_by_feature, sentiment_payload=sentiment_payload,
        gate=gate)
    return digest_assembler.assemble(final, date_str,
                                     holding_symbols=set(holding_syms))


def _send(body: str) -> None:
    from corvin_jarvis import channels
    channels.send_telegram(body)


def gather_inputs(*, portfolio_path: Path, universe_path: Path, db_path: Path,
                  geo_fetch, sentiment_fetch=None) -> dict:
    """portfolio.json + monitored_universe.json + daily_history → build_digest 입력."""
    holdings = []
    if portfolio_path.exists():
        pf = json.loads(portfolio_path.read_text())
        holdings = pf.get("holdings", [])
    universe = []
    if universe_path.exists():
        uni = json.loads(universe_path.read_text())
        universe = [t["symbol"] for t in uni.get("tickers", [])]

    backfill.init_db(db_path)
    syms = [h["symbol"] for h in holdings] + universe
    closes_by_sym = {s: [r["close"] for r in backfill.read_daily(db_path, s, 250)]
                     for s in syms}
    daily_by_feature = {f: backfill.read_daily(db_path, f, 500) for f in _FEATURES}
    stops = {}
    try:
        from corvin_jarvis.signal_engine import load_stops
        stops = load_stops()
    except Exception:
        stops = {}
    geo_payload = None
    try:
        geo_payload = geo_fetch()
    except Exception:
        geo_payload = None
    sentiment_payload = None
    if sentiment_fetch is not None:
        try:
            sentiment_payload = sentiment_fetch()
        except Exception:
            sentiment_payload = None
    return {"holdings": holdings, "universe": universe, "stops": stops,
            "closes_by_sym": closes_by_sym, "daily_by_feature": daily_by_feature,
            "geo_payload": geo_payload, "sentiment_payload": sentiment_payload}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--by-model", action="store_true",
                    help="모델별 상세 뷰(게이트 무관 전부 표시, 탈락 태깅)")
    args = ap.parse_args(argv)

    date_str = datetime.now(KST).strftime("%Y-%m-%d")
    backfill.init_db(_DB)
    _PF = Path(__file__).resolve().parent.parent.parent / "portfolio.json"
    _UNI = Path(__file__).resolve().parent.parent / "monitored_universe.json"
    inp = gather_inputs(portfolio_path=_PF, universe_path=_UNI, db_path=_DB,
                        geo_fetch=lambda: None)
    if args.by_model:
        gate = backtest.load_gate()
        results, _ = build_results(
            holdings=inp["holdings"], universe=inp["universe"],
            geo_payload=inp["geo_payload"], stops=inp["stops"],
            closes_by_sym=inp["closes_by_sym"],
            daily_by_feature=inp["daily_by_feature"],
            sentiment_payload=inp.get("sentiment_payload"),
            gate=gate, respect_gate=False)
        text = digest_assembler.by_model(results, date_str, gate=gate)
    else:
        text = build_digest(date_str=date_str, holdings=inp["holdings"],
                            universe=inp["universe"], db_path=_DB,
                            geo_payload=inp["geo_payload"], stops=inp["stops"],
                            closes_by_sym=inp["closes_by_sym"],
                            daily_by_feature=inp["daily_by_feature"],
                            sentiment_payload=inp.get("sentiment_payload"))
    if args.dry_run or args.by_model:
        print(text)
    else:
        _send(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
