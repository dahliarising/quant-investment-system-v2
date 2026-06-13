"""portfolio.json.bak.* 백업들에서 손익률(%) 시계열을 만든다 (관대한 파싱).

설계 A: 절대 총자산이 아니라 평단 대비 실현 손익률(%)을 추적한다 — 입금/매수에
영향받지 않는 순수 수익률 곡선. 같은 날짜의 백업이 여러 개면 가장 늦은
타임스탬프를 채택하고(중복 제거), 현재 portfolio.json은 자기 날짜에서 우선한다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 프로젝트 루트
# (날짜, 나머지 접미사) — 접미사는 같은 날짜 내 최신 백업 판별용 정렬 키.
_DATE_RE = re.compile(r"portfolio\.json\.bak\.(\d{4}-\d{2}-\d{2})(.*)$")


def _extract_pnl_pct(data: dict[str, Any]) -> float | None:
    totals = data.get("totals") or {}
    for key in ("equityPnlPct", "pnlPct"):
        v = totals.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_series(root: Path = BASE_DIR) -> list[dict[str, Any]]:
    # date -> (정렬키, pnl_pct). 같은 날짜는 더 큰 정렬키(늦은 타임스탬프)가 이긴다.
    by_date: dict[str, tuple[str, float]] = {}
    for f in root.glob("portfolio.json.bak.*"):
        m = _DATE_RE.match(f.name)
        if not m:
            continue
        date, suffix = m.group(1), m.group(2)
        data = _load(f)
        if not data:
            continue
        pct = _extract_pnl_pct(data)
        if pct is None:
            continue
        if date not in by_date or suffix > by_date[date][0]:
            by_date[date] = (suffix, pct)
    # 현재 portfolio.json — 자기 날짜에서 항상 우선('~'은 어떤 타임스탬프보다 큼).
    cur = _load(root / "portfolio.json")
    if cur:
        pct = _extract_pnl_pct(cur)
        if pct is not None:
            by_date[cur.get("updatedAt", "now")] = ("~", pct)
    return [{"date": d, "pnl_pct": round(by_date[d][1], 2)} for d in sorted(by_date)]
