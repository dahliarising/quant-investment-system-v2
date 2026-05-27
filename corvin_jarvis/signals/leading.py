"""선행 신호 (Phase B) — 상대강도(RS): 종목이 시장 지수보다 얼마나 센가/약한가."""
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


def check_relative_strength(
    universe_entries: list[dict[str, Any]],
    index_pct: float | None,
    config: dict[str, Any],
    phase: str,
    index_name: str = "",
) -> list[dict[str, Any]]:
    """종목 pct_change 와 벤치마크 지수 pct_change 의 차이(RS, %p)가 임계 초과 시 alert."""
    if index_pct is None:
        return []
    cfg = config.get("leading", {})
    rs_min = float(cfg.get("rs_min_pct", 2.0))
    emoji, label = _PHASE_EMOJI.get(phase, ""), _PHASE_LABEL.get(phase, phase)
    bench = index_name or "지수"

    alerts: list[dict[str, Any]] = []
    for entry in universe_entries:
        pct = entry.get("pct_change")
        if pct is None:
            continue
        rs = pct - index_pct
        if abs(rs) < rs_min:
            continue
        sev = _classify_severity(rs, rs_min)
        direction = "강세" if rs > 0 else "약세"
        sym = entry.get("symbol")
        alerts.append({
            "category": "leading_rs",
            "metric": f"rs_{sym}_{phase}",
            "severity": sev.value if isinstance(sev, Severity) else str(sev),
            "message": (f"{emoji} {_name_tag(entry)} 상대강도 {direction} {rs:+.2f}%p "
                        f"(종목 {pct:+.2f}% vs {bench} {index_pct:+.2f}%, 임계 ±{rs_min}%p, {label})"),
            "value": round(rs, 4),
            "threshold": rs_min,
            "phase": phase,
        })
    return alerts
