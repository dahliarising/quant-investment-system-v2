"""portfolio.json.bak.* 백업들에서 총자산 시계열을 만든다 (관대한 파싱)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 프로젝트 루트
_DATE_RE = re.compile(r"portfolio\.json\.bak\.(\d{4}-\d{2}-\d{2})")


def _extract_total(data: dict[str, Any]) -> float | None:
    totals = data.get("totals") or {}
    for key in ("totalAssetsKRW", "valueKRW", "equityValueKRW"):
        v = totals.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    holdings = data.get("holdings")
    if isinstance(holdings, list):
        s = sum(h.get("valueKRW", 0) for h in holdings if isinstance(h.get("valueKRW"), (int, float)))
        if s > 0:
            return float(s)
    return None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_series(root: Path = BASE_DIR) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for f in root.glob("portfolio.json.bak.*"):
        m = _DATE_RE.match(f.name)
        if not m:
            continue
        data = _load(f)
        if not data:
            continue
        total = _extract_total(data)
        if total is None:
            continue
        points.append({"date": m.group(1), "value": round(total)})
    cur = _load(root / "portfolio.json")
    if cur:
        total = _extract_total(cur)
        if total is not None:
            points.append({"date": cur.get("updatedAt", "now"), "value": round(total)})
    points.sort(key=lambda p: p["date"])
    return points
