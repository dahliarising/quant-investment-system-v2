"""monitored_universe.json 로드·검증."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

BASE_DIR: Final = Path(__file__).resolve().parent.parent
UNIVERSE_FILE: Final = BASE_DIR / "monitored_universe.json"


@dataclass(frozen=True)
class MonitoredTicker:
    symbol: str
    market: str   # "KR" | "US"
    sector: str
    name: str


def load(path: Path = UNIVERSE_FILE) -> list[MonitoredTicker]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[MonitoredTicker] = []
    for t in data.get("tickers", []):
        out.append(MonitoredTicker(
            symbol=str(t["symbol"]),
            market=str(t["market"]),
            sector=str(t["sector"]),
            name=str(t.get("name", t["symbol"])),
        ))
    return out


def by_sector(tickers: list[MonitoredTicker]) -> dict[str, list[MonitoredTicker]]:
    groups: dict[str, list[MonitoredTicker]] = defaultdict(list)
    for t in tickers:
        groups[t.sector].append(t)
    return dict(groups)
