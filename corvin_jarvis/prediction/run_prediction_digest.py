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

from corvin_jarvis.prediction import (backfill, digest_assembler, m_geopolitical,
                                      m_momentum, m_probability, m_velocity, m_vector)
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


def build_digest(*, date_str, holdings, universe, db_path, geo_payload,
                 stops, closes_by_sym, daily_by_feature) -> str:
    results: list[PredictionResult] = []
    results += _safe("velocity", _run_velocity, holdings, stops, closes_by_sym)
    for pos in holdings:
        sym = str(pos.get("symbol", ""))
        results += _safe("probability", m_probability.run_symbol, sym,
                         [c for c in closes_by_sym.get(sym, [])], stops.get(sym, 0.0))
    for sym in universe:
        results += _safe("momentum", m_momentum.run_symbol, sym,
                         closes_by_sym.get(sym, []))
    results += _safe("geopolitical", m_geopolitical.run, geo_payload)
    results += _safe("vector_analog", m_vector.predict, daily_by_feature,
                     features=_FEATURES)
    return digest_assembler.assemble(results, date_str)


def _send(body: str) -> None:
    from corvin_jarvis import channels
    channels.send_telegram(body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    date_str = datetime.now(KST).strftime("%Y-%m-%d")
    backfill.init_db(_DB)
    # 실제 데이터 수집은 운영에서 portfolio/universe/MCP로 채운다.
    # (여기서는 골격 — 세부 수집 로직은 운영 진입점에서 주입)
    text = build_digest(date_str=date_str, holdings=[], universe=[], db_path=_DB,
                        geo_payload=None, stops={}, closes_by_sym={},
                        daily_by_feature={})
    if args.dry_run:
        print(text)
    else:
        _send(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
