"""Corvin Jarvis — Pulse (Phase 1)

매 시간 cron으로 호출되어 시장/원자재/FX/포트폴리오 snapshot을 state/에 저장.
순수 Python only — MCP 의존 없음 (cron 환경에서 작동해야 함).

사용:
    python corvin_jarvis/pulse.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf

# scripts/kr_data.py 재사용
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kr_data import get_kr_index_data, get_kr_stock_data, is_korean_ticker  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc

STATE_DIR = BASE_DIR / "state"
SNAPSHOTS_DIR = STATE_DIR / "snapshots"
LATEST_FILE = STATE_DIR / "latest.json"
PORTFOLIO_FILE = PROJECT_ROOT / "portfolio.json"
LOG_FILE = STATE_DIR / "pulse.log"
CONFIG_FILE = BASE_DIR / "config.json"
TIMESERIES_DB = STATE_DIR / "timeseries.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("corvin.pulse")


@dataclass(frozen=True)
class Quote:
    price: float | None
    pct_change: float | None
    source: str
    error: str | None = None


@dataclass(frozen=True)
class PositionQuote:
    symbol: str
    shares: float
    avg_price: float
    currency: str
    current_price: float | None
    pnl_pct: float | None
    market_value: float | None
    error: str | None = None


@dataclass
class Snapshot:
    timestamp_utc: str
    timestamp_kst: str
    indices: dict[str, dict[str, Any]] = field(default_factory=dict)
    commodities: dict[str, dict[str, Any]] = field(default_factory=dict)
    fx: dict[str, dict[str, Any]] = field(default_factory=dict)
    portfolio: list[dict[str, Any]] = field(default_factory=list)
    portfolio_summary: dict[str, Any] = field(default_factory=dict)


def _yf_quote(ticker: str) -> Quote:
    """yfinance로 2일 history fetch해서 last + pct_change."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return Quote(price=None, pct_change=None, source="yfinance", error="no data")
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last
        pct = (last / prev - 1) * 100 if prev else 0.0
        return Quote(price=round(last, 4), pct_change=round(pct, 2), source="yfinance")
    except Exception as e:  # noqa: BLE001
        return Quote(price=None, pct_change=None, source="yfinance", error=str(e))


def fetch_indices() -> dict[str, dict[str, Any]]:
    """미국 지수는 yfinance, 한국 지수는 FinanceDataReader (kr_data.py 활용)."""
    out: dict[str, dict[str, Any]] = {}

    # 미국 지수
    us_indices = {"sp500": "^GSPC", "nasdaq": "^IXIC", "dow": "^DJI", "vix": "^VIX"}
    for name, tkr in us_indices.items():
        q = _yf_quote(tkr)
        out[name] = asdict(q)

    # 한국 지수 (pykrx/FDR로 강제)
    try:
        kospi = get_kr_index_data("KS11")
        out["kospi"] = {
            "price": kospi.get("현재가"),
            "pct_change": kospi.get("전일대비(%)"),
            "source": "FinanceDataReader",
            "error": kospi.get("에러"),
        }
    except Exception as e:  # noqa: BLE001
        out["kospi"] = {"price": None, "pct_change": None, "source": "FDR", "error": str(e)}

    try:
        kosdaq = get_kr_index_data("KQ11")
        out["kosdaq"] = {
            "price": kosdaq.get("현재가"),
            "pct_change": kosdaq.get("전일대비(%)"),
            "source": "FinanceDataReader",
            "error": kosdaq.get("에러"),
        }
    except Exception as e:  # noqa: BLE001
        out["kosdaq"] = {"price": None, "pct_change": None, "source": "FDR", "error": str(e)}

    return out


def fetch_commodities() -> dict[str, dict[str, Any]]:
    tickers = {
        "brent": "BZ=F",
        "wti": "CL=F",
        "gold": "GC=F",
        "silver": "SI=F",
        "copper": "HG=F",
    }
    return {name: asdict(_yf_quote(tkr)) for name, tkr in tickers.items()}


def fetch_fx() -> dict[str, dict[str, Any]]:
    pairs = {
        "usd_krw": "KRW=X",
        "eur_usd": "EURUSD=X",
        "jpy_krw": "JPYKRW=X",
        "dxy": "DX-Y.NYB",
    }
    return {name: asdict(_yf_quote(tkr)) for name, tkr in pairs.items()}


def load_watchlist(config_path: Path = CONFIG_FILE) -> list[str]:
    """config.json의 watchlist 리스트 반환. 누락/없음 → 빈 리스트."""
    if not config_path.exists():
        return []
    try:
        data = json.loads(config_path.read_text())
        items = data.get("watchlist", [])
        return [str(s) for s in items] if isinstance(items, list) else []
    except (json.JSONDecodeError, OSError) as e:
        log.warning("watchlist load failed: %s", e)
        return []


def fetch_watchlist(symbols: list[str]) -> list[dict[str, Any]]:
    """holdings와 동일한 quote 인터페이스로 watchlist 가격 수집."""
    out: list[dict[str, Any]] = []
    for sym in symbols:
        try:
            if is_korean_ticker(sym):
                data = get_kr_stock_data(sym)
                if "에러" in data:
                    out.append({"symbol": sym, "price": None, "pct_change": None,
                                "source": "FinanceDataReader", "error": data["에러"]})
                else:
                    out.append({"symbol": sym, "price": float(data["현재가"]),
                                "pct_change": float(data.get("전일대비(%)", 0.0)),
                                "source": "FinanceDataReader", "error": None})
            else:
                q = _yf_quote(sym)
                out.append({"symbol": sym, "price": q.price, "pct_change": q.pct_change,
                            "source": q.source, "error": q.error})
        except Exception as e:  # noqa: BLE001
            out.append({"symbol": sym, "price": None, "pct_change": None,
                        "source": "unknown", "error": str(e)})
    return out


def _fetch_position_quote(holding: dict[str, Any]) -> PositionQuote:
    sym = holding["symbol"]
    shares = float(holding["shares"])
    avg = float(holding["avgPrice"])
    cur = holding.get("currency", "USD")

    try:
        if is_korean_ticker(sym):
            data = get_kr_stock_data(sym)
            if "에러" in data:
                return PositionQuote(
                    symbol=sym, shares=shares, avg_price=avg, currency=cur,
                    current_price=None, pnl_pct=None, market_value=None, error=data["에러"],
                )
            price = float(data["현재가"])
        else:
            q = _yf_quote(sym)
            if q.error or q.price is None:
                return PositionQuote(
                    symbol=sym, shares=shares, avg_price=avg, currency=cur,
                    current_price=None, pnl_pct=None, market_value=None, error=q.error,
                )
            price = q.price

        pnl = (price / avg - 1) * 100 if avg else 0.0
        return PositionQuote(
            symbol=sym, shares=shares, avg_price=avg, currency=cur,
            current_price=round(price, 4), pnl_pct=round(pnl, 2),
            market_value=round(price * shares, 2),
        )
    except Exception as e:  # noqa: BLE001
        return PositionQuote(
            symbol=sym, shares=shares, avg_price=avg, currency=cur,
            current_price=None, pnl_pct=None, market_value=None, error=str(e),
        )


def fetch_portfolio() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not PORTFOLIO_FILE.exists():
        log.warning("portfolio.json 없음 — skip")
        return [], {}

    portfolio_data = json.loads(PORTFOLIO_FILE.read_text())
    holdings = portfolio_data.get("holdings", [])

    quotes = [_fetch_position_quote(h) for h in holdings]
    position_dicts = [asdict(q) for q in quotes]

    total_krw = sum(
        q.market_value for q in quotes
        if q.currency == "KRW" and q.market_value is not None
    )
    total_usd = sum(
        q.market_value for q in quotes
        if q.currency == "USD" and q.market_value is not None
    )

    winners = [q for q in quotes if q.pnl_pct is not None and q.pnl_pct > 0]
    losers = [q for q in quotes if q.pnl_pct is not None and q.pnl_pct < 0]

    summary = {
        "total_value_krw": round(total_krw, 2),
        "total_value_usd": round(total_usd, 2),
        "position_count": len(quotes),
        "winners_count": len(winners),
        "losers_count": len(losers),
        "best_pnl_pct": max((q.pnl_pct for q in quotes if q.pnl_pct is not None), default=None),
        "worst_pnl_pct": min((q.pnl_pct for q in quotes if q.pnl_pct is not None), default=None),
        "stale_as_of": portfolio_data.get("updatedAt"),
    }
    return position_dicts, summary


def run_pulse() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(UTC)
    now_kst = now_utc.astimezone(KST)
    log.info("Pulse 시작 — %s KST", now_kst.strftime("%Y-%m-%d %H:%M:%S"))

    snapshot = Snapshot(
        timestamp_utc=now_utc.isoformat(),
        timestamp_kst=now_kst.isoformat(),
    )

    log.info("Fetching indices...")
    snapshot.indices = fetch_indices()
    log.info("Fetching commodities...")
    snapshot.commodities = fetch_commodities()
    log.info("Fetching FX...")
    snapshot.fx = fetch_fx()
    log.info("Fetching portfolio quotes...")
    snapshot.portfolio, snapshot.portfolio_summary = fetch_portfolio()

    snapshot_path = SNAPSHOTS_DIR / f"{now_kst.strftime('%Y%m%d-%H%M')}.json"
    snapshot_dict = asdict(snapshot)
    snapshot_path.write_text(json.dumps(snapshot_dict, indent=2, default=str, ensure_ascii=False))
    LATEST_FILE.write_text(json.dumps(snapshot_dict, indent=2, default=str, ensure_ascii=False))

    log.info("Pulse 완료 — %s", snapshot_path.name)
    return snapshot_path


def _print_summary(snapshot_path: Path) -> None:
    data = json.loads(snapshot_path.read_text())
    log.info("=== Pulse Summary ===")
    for name, q in data["indices"].items():
        log.info("  [IDX] %-8s %s  (%+.2f%%)", name, q.get("price"), q.get("pct_change") or 0)
    for name, q in data["commodities"].items():
        log.info("  [CMD] %-8s %s  (%+.2f%%)", name, q.get("price"), q.get("pct_change") or 0)
    for name, q in data["fx"].items():
        log.info("  [FX]  %-8s %s  (%+.2f%%)", name, q.get("price"), q.get("pct_change") or 0)
    log.info("  [PFL] positions=%s value_KRW=%s value_USD=%s",
             data["portfolio_summary"].get("position_count"),
             data["portfolio_summary"].get("total_value_krw"),
             data["portfolio_summary"].get("total_value_usd"))


if __name__ == "__main__":
    path = run_pulse()
    _print_summary(path)
