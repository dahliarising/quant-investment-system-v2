"""daily_history — 예측 모듈용 일봉 저장소 (intraday quote_history와 분리).

소스 규칙: KR 종목/지수 = FinanceDataReader/pykrx, US = FinanceDataReader.
yfinance는 한국 데이터 stale → KR에 절대 사용 금지.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_history (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    source TEXT,
    PRIMARY KEY (symbol, date)
);
"""


def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as c:
        c.execute(_SCHEMA)


def upsert_rows(db_path: Path, rows: list[dict[str, Any]]) -> int:
    sql = ("INSERT INTO daily_history (symbol,date,open,high,low,close,volume,source) "
           "VALUES (:symbol,:date,:open,:high,:low,:close,:volume,:source) "
           "ON CONFLICT(symbol,date) DO UPDATE SET "
           "open=excluded.open,high=excluded.high,low=excluded.low,"
           "close=excluded.close,volume=excluded.volume,source=excluded.source")
    with sqlite3.connect(db_path) as c:
        c.executemany(sql, rows)
        return c.total_changes


def read_daily(db_path: Path, symbol: str, lookback: int = 250) -> list[dict[str, Any]]:
    """최근 lookback개 일봉을 시간순(오름차순)으로."""
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(
            "SELECT * FROM (SELECT * FROM daily_history WHERE symbol=? "
            "ORDER BY date DESC LIMIT ?) ORDER BY date ASC", (symbol, lookback))
        return [dict(r) for r in cur.fetchall()]


def last_date(db_path: Path, symbol: str) -> str | None:
    with sqlite3.connect(db_path) as c:
        row = c.execute("SELECT MAX(date) FROM daily_history WHERE symbol=?",
                        (symbol,)).fetchone()
        return row[0] if row and row[0] else None


Fetcher = Callable[[str, str, str], list[dict[str, Any]]]


def _num(v: Any, cast: Callable[[float], Any] = float) -> Any:
    """NaN/None-safe 숫자 변환. 지수·FX·선물은 Volume이 NaN인 행이 있어
    int(NaN) ValueError 폭발 → incremental_update가 swallow → 일봉 누락 방지."""
    try:
        if v is None:
            return cast(0)
        f = float(v)
        if math.isnan(f):
            return cast(0)
        return cast(f)
    except (TypeError, ValueError):
        return cast(0)


def _row_from(symbol: str, date: str, open: Any, high: Any, low: Any,
              close: Any, volume: Any, source: str) -> dict[str, Any]:
    """순수 row 빌더 — 모든 숫자 컬럼에 _num 적용 (NaN-safe). 단위 테스트 가능."""
    return {"symbol": symbol, "date": date,
            "open": _num(open, float), "high": _num(high, float),
            "low": _num(low, float), "close": _num(close, float),
            "volume": _num(volume, int), "source": source}


def _default_fetch(symbol: str, market: str, start: str) -> list[dict[str, Any]]:
    """FinanceDataReader 일봉 → daily_history row. KR/US 동일 API."""
    import FinanceDataReader as fdr
    df = fdr.DataReader(symbol, start)
    out: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        out.append(_row_from(symbol, idx.strftime("%Y-%m-%d"),
                             row.get("Open"), row.get("High"), row.get("Low"),
                             row.get("Close"), row.get("Volume"), "fdr"))
    return out


def incremental_update(db_path: Path, symbols: list[tuple[str, str]],
                       fetcher: Fetcher | None = None) -> int:
    """symbols = [(symbol, market)]. last_date 다음날부터 fetch 후 upsert."""
    fetch = fetcher or _default_fetch
    total = 0
    for symbol, market in symbols:
        last = last_date(db_path, symbol)
        if last:
            start = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start = "2016-01-01"  # 초기 backfill ~10년
        try:
            rows = fetch(symbol, market, start)
        except Exception:
            continue  # 네트워크 실패 → 기존 캐시로 진행
        if rows:
            total += upsert_rows(db_path, rows)
    return total
