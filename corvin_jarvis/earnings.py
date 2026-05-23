"""Corvin Jarvis — Earnings Calendar (Tier 1.3)

yfinance.Ticker.calendar로 어닝 발표일 수집 → SQLite 저장 → D-7/D-3/D-1 alert.

기존 timeseries.db에 별도 테이블 earnings_calendar로 저장.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.earnings")

SCHEMA = """
CREATE TABLE IF NOT EXISTS earnings_calendar (
    symbol TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    eps_avg REAL,
    revenue_avg REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, earnings_date)
);
CREATE INDEX IF NOT EXISTS idx_ec_symbol_date ON earnings_calendar (symbol, earnings_date);
"""

ALERT_OFFSETS = (7, 3, 1)


def init_earnings_table(db_path: Path) -> None:
    """earnings_calendar 테이블 생성 (idempotent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
    log.info("earnings_calendar table ready at %s", db_path)


def upsert_earnings(
    db_path: Path,
    symbol: str,
    dates: list[date],
    eps_avg: float | None = None,
    revenue_avg: float | None = None,
) -> int:
    """주어진 (symbol, date) 쌍 upsert. 동일 PK는 UPDATE. 반환: row 수."""
    init_earnings_table(db_path)
    if not dates:
        return 0
    now_iso = datetime.now().isoformat(timespec="seconds")
    rows = [
        (symbol, d.isoformat(), eps_avg, revenue_avg, now_iso)
        for d in dates
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO earnings_calendar
                 (symbol, earnings_date, eps_avg, revenue_avg, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(symbol, earnings_date) DO UPDATE SET
                 eps_avg = excluded.eps_avg,
                 revenue_avg = excluded.revenue_avg,
                 updated_at = excluded.updated_at""",
            rows,
        )
    return len(rows)


def _yf_ticker_calendar(symbol: str) -> dict[str, Any]:
    """yfinance.Ticker(symbol).calendar wrapper (mock 가능하도록 분리)."""
    import yfinance as yf
    cal = yf.Ticker(symbol).calendar
    return cal if isinstance(cal, dict) else {}


def fetch_earnings_date(symbol: str) -> dict[str, Any]:
    """yfinance에서 어닝 정보 fetch. 실패 시 dates=[] + error 메시지."""
    try:
        cal = _yf_ticker_calendar(symbol)
        raw_dates = cal.get("Earnings Date") or []
        parsed_dates: list[date] = []
        for d in raw_dates:
            if isinstance(d, date):
                parsed_dates.append(d)
            elif isinstance(d, str):
                try:
                    parsed_dates.append(date.fromisoformat(d))
                except ValueError:
                    continue
        return {
            "symbol": symbol,
            "dates": parsed_dates,
            "eps_avg": cal.get("Earnings Average"),
            "revenue_avg": cal.get("Revenue Average"),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "dates": [], "eps_avg": None,
                "revenue_avg": None, "error": str(e)}


def pending_earnings(
    db_path: Path,
    today: date,
    days_ahead: int = 14,
) -> list[dict[str, Any]]:
    """today 이상 ~ today+days_ahead 이내의 어닝 row 반환 (오름차순)."""
    if not db_path.exists():
        return []
    end = today + timedelta(days=days_ahead)
    sql = (
        "SELECT symbol, earnings_date, eps_avg, revenue_avg "
        "FROM earnings_calendar "
        "WHERE earnings_date >= ? AND earnings_date <= ? "
        "ORDER BY earnings_date ASC"
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, (today.isoformat(), end.isoformat())).fetchall()
    return [dict(r) for r in rows]


def _severity_for_offset(offset_days: int) -> str:
    """D-7 → medium, D-3/D-1 → high."""
    return "high" if offset_days <= 3 else "medium"


def refresh_earnings_calendar(
    db_path: Path,
    symbols: list[str],
) -> dict[str, Any]:
    """모든 symbols에 대해 fetch + upsert. errors는 별도 수집."""
    fetched = 0
    errors: list[dict[str, str]] = []
    for sym in symbols:
        result = fetch_earnings_date(sym)
        if result["error"]:
            errors.append({"symbol": sym, "error": result["error"]})
            continue
        if not result["dates"]:
            continue
        upsert_earnings(
            db_path, sym, result["dates"],
            eps_avg=result.get("eps_avg"),
            revenue_avg=result.get("revenue_avg"),
        )
        fetched += 1
    return {"fetched": fetched, "errors": errors}


def build_earnings_alerts(
    db_path: Path,
    today: date,
) -> list[dict[str, Any]]:
    """D-7/D-3/D-1 어닝 alert 생성. 기존 compare.py Alert 포맷 호환."""
    alerts: list[dict[str, Any]] = []
    rows = pending_earnings(db_path, today=today, days_ahead=max(ALERT_OFFSETS))
    for r in rows:
        edate = date.fromisoformat(r["earnings_date"])
        delta = (edate - today).days
        if delta not in ALERT_OFFSETS:
            continue
        sev = _severity_for_offset(delta)
        eps = r.get("eps_avg")
        eps_str = f" EPS est ${eps:.2f}" if eps is not None else ""
        alerts.append({
            "category": "earnings",
            "metric": r["symbol"],
            "severity": sev,
            "message": f"{r['symbol']} 어닝 D-{delta} ({edate.isoformat()}){eps_str}",
            "value": delta,
            "threshold": None,
            "delta_from_prev": None,
        })
    return alerts
