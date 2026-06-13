# corvin_jarvis/prediction/seed_phase2_backtest.py
"""Phase 2 백테스트 게이트 생성 — daily_history walk-forward → phase2_backtest.json.

오프라인 실행(매일 다이제스트와 분리). 통과 모델만 다이제스트에 합류한다.
재실행 가이드: 데이터 추가 적재 후 주기적으로 갱신.
"""
import json
from pathlib import Path

from corvin_jarvis.prediction import backtest

_DB = Path(__file__).resolve().parent.parent / "state" / "daily_history.db"
_PF = Path(__file__).resolve().parent.parent.parent / "portfolio.json"
_FEATURES = ["kospi", "nasdaq", "vix", "usd_krw", "gold", "copper", "dxy"]


def _holdings() -> list[str]:
    if not _PF.exists():
        return []
    pf = json.loads(_PF.read_text())
    return [h["symbol"] for h in pf.get("holdings", []) if h.get("symbol")]


def main() -> None:
    # montecarlo는 보유종목에 적용 → 보유종목으로 게이트 검증(정직)
    gate = backtest.run_all(_DB, features=_FEATURES, holdings=_holdings())
    backtest.save_gate(gate)
    print(f"백테스트 게이트 저장: {backtest.GATE_PATH}")
    for sys_name, res in gate.items():
        mark = "✅ 통과" if res.get("passed") else "❌ 탈락"
        metric = (f"hit {res.get('hit_rate', 0):.3f} vs base {res.get('baseline', 0):.3f}"
                  if "hit_rate" in res
                  else f"coverage {res.get('coverage', 0):.3f} (target {res.get('target', 0)})")
        print(f"  {sys_name:12} {mark}  {metric}  n={res.get('n')}")


if __name__ == "__main__":
    main()
