"""감시 universe의 종목별 급등락 + 섹터 바스켓 alert 생성."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from compare import Severity, _classify_severity  # noqa: E402

_PHASE_EMOJI = {"provisional": "🟡", "confirmed": "✅"}
_PHASE_LABEL = {"provisional": "잠정", "confirmed": "확정"}


def _name_tag(entry: dict[str, Any]) -> str:
    name = entry.get("name")
    sym = entry.get("symbol")
    return f"{name}({sym})" if name else str(sym)


def check_tickers(snapshot: dict[str, Any], config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    cfg = config.get("stock_pct_change", {})
    default_th = float(cfg.get("default", 5.0))
    overrides = cfg.get("overrides", {})
    emoji, label = _PHASE_EMOJI.get(phase, ""), _PHASE_LABEL.get(phase, phase)

    alerts: list[dict[str, Any]] = []
    for entry in snapshot.get("universe", []):
        sym = entry.get("symbol")
        pct = entry.get("pct_change")
        if pct is None:
            continue
        th = float(overrides.get(sym, default_th))
        if abs(pct) < th:
            continue
        sev = _classify_severity(pct, th)
        direction = "급등" if pct > 0 else "급락"
        alerts.append({
            "category": "universe",
            "metric": f"universe_{sym}_{phase}",
            "severity": sev.value if isinstance(sev, Severity) else str(sev),
            "message": f"{emoji} {_name_tag(entry)} {direction} {pct:+.2f}% (임계 ±{th}%, {label})",
            "value": pct,
            "threshold": th,
            "phase": phase,
        })
    return alerts
