"""플레이북 데이터 모델 (불변)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Technicals:
    symbol: str
    market: str  # "US" | "KR"
    price: float
    ma20: float
    ma50: float
    rsi: float
    hi_52w: float


@dataclass(frozen=True)
class Zone:
    kind: str    # "buy" | "trim"
    label: str
    ratio: int   # percent
    low: float
    high: float
    note: str


@dataclass(frozen=True)
class Playbook:
    symbol: str
    name: str
    stance: str           # "ENTER" | "ACCUMULATE" | "HARVEST"
    tech: Technicals
    zones: tuple[Zone, ...]
    status: str           # "BUY_NOW" | "TRIM_NOW" | "WAIT" | "INVALID"
    badge: str
    pnl_pct: float | None
    active_zone: Zone | None
