"""arbiter 입력 어댑터 — 엔진별 신호를 공통 normalize dict로 + CLI.

CLI(`python3 -m corvin_jarvis.signals.arbiter_inputs`)는 라이브 신호를 수집해
arbitrate 후 state/final_actions.json에 기록 (digest가 읽음).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from corvin_jarvis.signals import arbiter

KST = ZoneInfo("Asia/Seoul")
STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "final_actions.json"

_ENGINE_INTENT = {"STOP": "defensive", "WATCH": "warn", "TRIM": "trim", "HOLD": "hold"}
_JARVIS_DEFENSIVE_KINDS = {"stop_loss", "hardstop"}
# ledger open에서 중재에 쓰는 엔진 — 라이브 평가가 없는 엔진만 (이중 계상 방지)
_LEDGER_ENGINES = {"leading", "jarvis"}


def _row(engine: str, symbol: str, kind: str, intent: str,
         urgency: int, confidence: float | None, note: str) -> dict[str, Any]:
    return {"engine": engine, "symbol": symbol, "kind": kind, "intent": intent,
            "urgency": urgency, "confidence": confidence, "note": note}


def normalize_engine_signals(sigs: list[dict]) -> list[dict]:
    """signal_engine EngineSignal.to_dict() 리스트 → 공통 dict. UNKNOWN 제외."""
    out = []
    for s in sigs:
        intent = _ENGINE_INTENT.get(str(s.get("kind", "")))
        if intent is None:
            continue
        out.append(_row("signal_engine", str(s.get("symbol", "")), s["kind"], intent,
                        int(s.get("urgency") or 0), None, str(s.get("reason", ""))))
    return out


def normalize_predictive_signals(sigs: list[dict]) -> list[dict]:
    """predictive PredictiveSignal.to_dict() → 공통 dict. EVENT(매크로) 제외."""
    out = []
    for s in sigs:
        sym = str(s.get("symbol", ""))
        if not sym:
            continue
        out.append(_row("predictive", sym, str(s.get("kind", "")), "warn",
                        int(s.get("urgency") or 0), s.get("confidence"),
                        str(s.get("message", ""))))
    return out


def normalize_playbook_signals(sigs: list[dict]) -> list[dict]:
    """snapshot._build_signals 출력({sym,zone,stance,color}) → 공통 dict.

    color green=BUY_NOW, amber=TRIM_NOW (snapshot._build_signals의 매핑 역변환).
    """
    out = []
    for s in sigs:
        color = s.get("color")
        if color not in ("green", "amber"):
            continue  # 알 수 없는 색 → 추측 금지, 제외
        buy = color == "green"
        out.append(_row("playbook", str(s.get("sym", "")),
                        "BUY_NOW" if buy else "TRIM_NOW",
                        "buy" if buy else "trim",
                        50 if buy else 55, None, str(s.get("zone", ""))))
    return out


def normalize_ledger_open(rows: list[dict], dca_syms=frozenset()) -> list[dict]:
    """ledger open 행 → 공통 dict. leading/jarvis만 (라이브 엔진 이중 계상 방지).

    leading: direction bull→buy / bear→warn.
    jarvis: stop_loss·hardstop→defensive, 그 외 정보성 제외 (방향 없음).
    매크로(symbol="") 제외. dca_syms의 방어(stop) 행은 제외 — 가격손절 면제 일관성.
    """
    out = []
    for r in rows:
        engine = str(r.get("engine", ""))
        sym = str(r.get("symbol", ""))
        if engine not in _LEDGER_ENGINES or not sym:
            continue
        kind = str(r.get("kind", ""))
        note = str((r.get("evidence") or {}).get("message", ""))
        conf = r.get("confidence")
        if engine == "leading":
            d = r.get("direction")
            if d == "bull":
                out.append(_row(engine, sym, kind, "buy",
                                int(r.get("urgency") or 50), conf, note))
            elif d == "bear":
                out.append(_row(engine, sym, kind, "warn",
                                int(r.get("urgency") or 50), conf, note))
            continue
        if kind in _JARVIS_DEFENSIVE_KINDS and sym not in dca_syms:
            out.append(_row(engine, sym, kind, "defensive",
                            int(r.get("urgency") or 90), conf, note))
    return out


def _summarize_predictive(pred_sigs: list[dict]) -> list[dict]:
    """digest용 예측 요약 — EVENT 우선, 그 외 urgency 내림차순, cap 5."""
    ranked = sorted(pred_sigs or [],
                    key=lambda s: (0 if s.get("kind") == "EVENT" else 1,
                                   -(s.get("urgency") or 0)))
    return [{"kind": s.get("kind"), "symbol": s.get("symbol") or "",
             "message": s.get("message", ""), "horizon_days": s.get("horizon_days"),
             "urgency": s.get("urgency")}
            for s in ranked[:5]]


def build_final_actions(*, engine_sigs: list[dict], pred_sigs: list[dict],
                        playbook_sigs: list[dict], ledger_open: list[dict],
                        calibration: dict | None,
                        dca_syms=frozenset(),
                        out_path: Path | None = None,
                        now: datetime | None = None) -> dict[str, Any]:
    """normalize → arbitrate → {ts, actions} 반환 + state 파일 기록."""
    signals = (normalize_engine_signals(engine_sigs)
               + normalize_predictive_signals(pred_sigs)
               + normalize_playbook_signals(playbook_sigs)
               + normalize_ledger_open(ledger_open, dca_syms))
    actions = [a.to_dict() for a in arbiter.arbitrate(signals, calibration=calibration)]
    t = now or datetime.now(KST)
    if t.tzinfo is None:
        t = t.replace(tzinfo=KST)
    result = {"ts": t.isoformat(timespec="seconds"), "actions": actions,
              "predictive": _summarize_predictive(pred_sigs)}
    target = out_path or STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def collect_live() -> dict[str, Any]:
    """라이브 수집 (snapshot 헬퍼 재사용) → build_final_actions. CLI 전용.

    snapshot 프라이빗 헬퍼(_load_portfolio/_positions/_held_for_engine/
    _engine_signals/_predictive_signals/_build_signals)에 의존 — snapshot 내부
    변경 시 이 함수도 함께 갱신할 것. 각 단계는 _safe 격리: 한 엔진 실패가
    전체 중재를 막지 않고 빈 입력으로 강등된다.
    """
    from corvin_jarvis.dashboard import snapshot as snap
    from corvin_jarvis.playbook import builder
    from corvin_jarvis.signals import calibration as cal_mod
    from corvin_jarvis.signals import ledger

    pf = snap._load_portfolio()
    positions = snap._safe(lambda: snap._positions(pf), [])
    held = snap._safe(lambda: snap._held_for_engine(pf, positions), [])
    engine_sigs = snap._safe(lambda: snap._engine_signals(held), [])
    pred_sigs = snap._safe(lambda: snap._predictive_signals(held), [])
    holdings = snap._safe(builder.load_holdings, {})
    playbook_sigs = snap._safe(lambda: snap._build_signals(holdings), [])
    ledger_open = snap._safe(ledger.fetch_open, [])
    calibration = snap._safe(cal_mod.compute, None)
    dca_syms = frozenset(h["symbol"] for h in held
                         if str(h.get("bucket")) == "dca")
    return build_final_actions(engine_sigs=engine_sigs, pred_sigs=pred_sigs,
                               playbook_sigs=playbook_sigs, ledger_open=ledger_open,
                               calibration=calibration, dca_syms=dca_syms)


if __name__ == "__main__":
    res = collect_live()
    print(json.dumps({"actions": len(res["actions"]), "ts": res["ts"]}, ensure_ascii=False))
