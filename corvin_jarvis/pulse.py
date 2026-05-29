"""Corvin Jarvis — Pulse (Phase 1)

매 시간 cron으로 호출되어 시장/원자재/FX/포트폴리오 snapshot을 state/에 저장.
시세는 quote_provider 어댑터를 통해 KIS 우선 + yfinance/pykrx 폴백.

사용:
    python corvin_jarvis/pulse.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# scripts/kr_data.py 재사용 (pulse → quote_provider → kr_data fallback)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from corvin_jarvis import quote_provider  # noqa: E402
from kr_data import is_korean_ticker  # noqa: E402

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
class PositionQuote:
    symbol: str
    shares: float
    avg_price: float
    currency: str
    current_price: float | None
    pnl_pct: float | None
    market_value: float | None
    source: str = ""
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
    watchlist: list[dict[str, Any]] = field(default_factory=list)
    universe: list[dict[str, Any]] = field(default_factory=list)


def fetch_indices() -> dict[str, dict[str, Any]]:
    """US 지수 + KR 지수 모두 quote_provider로 통일."""
    out: dict[str, dict[str, Any]] = {}
    us_indices = {"sp500": "^GSPC", "nasdaq": "^IXIC", "dow": "^DJI", "vix": "^VIX"}
    for name, sym in us_indices.items():
        out[name] = quote_provider.get_us_index_quote(sym).to_dict()
    out["kospi"] = quote_provider.get_kr_index_quote("KS11").to_dict()
    out["kosdaq"] = quote_provider.get_kr_index_quote("KQ11").to_dict()
    return out


def fetch_commodities() -> dict[str, dict[str, Any]]:
    tickers = {
        "brent": "BZ=F",
        "wti": "CL=F",
        "gold": "GC=F",
        "silver": "SI=F",
        "copper": "HG=F",
    }
    return {name: quote_provider.get_commodity_quote(s).to_dict() for name, s in tickers.items()}


def fetch_fx() -> dict[str, dict[str, Any]]:
    pairs = {
        "usd_krw": "KRW=X",
        "eur_usd": "EURUSD=X",
        "jpy_krw": "JPYKRW=X",
        "dxy": "DX-Y.NYB",
    }
    return {name: quote_provider.get_fx_quote(s).to_dict() for name, s in pairs.items()}


def load_watchlist(config_path: Path = CONFIG_FILE) -> list[str]:
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
    out: list[dict[str, Any]] = []
    for sym in symbols:
        q = quote_provider.get_stock_quote(sym)
        out.append({
            "symbol": sym,
            "price": q.price,
            "pct_change": q.pct_change,
            "source": q.source,
            "error": q.error,
        })
    return out


def fetch_universe() -> list[dict[str, Any]]:
    """monitored_universe.json 종목 시세 fetch (alert 감지용)."""
    from corvin_jarvis.signals import universe_loader
    out: list[dict[str, Any]] = []
    for t in universe_loader.load():
        q = quote_provider.get_stock_quote(t.symbol)
        out.append({
            "symbol": t.symbol,
            "market": t.market,
            "sector": t.sector,
            "name": t.name,
            "price": q.price,
            "pct_change": q.pct_change,
            "source": q.source,
            "error": q.error,
        })
    return out


def _fetch_position_quote(holding: dict[str, Any]) -> PositionQuote:
    sym = holding["symbol"]
    shares = float(holding["shares"])
    cur = holding.get("currency", "USD")
    avg_key = "avgPriceKRW" if cur == "KRW" else "avgPriceUSD"
    avg = float(holding.get("avgPrice") or holding[avg_key])

    q = quote_provider.get_stock_quote(sym)
    if q.error or q.price is None:
        return PositionQuote(
            symbol=sym, shares=shares, avg_price=avg, currency=cur,
            current_price=None, pnl_pct=None, market_value=None,
            source=q.source, error=q.error,
        )

    price = q.price
    pnl = (price / avg - 1) * 100 if avg else 0.0
    return PositionQuote(
        symbol=sym, shares=shares, avg_price=avg, currency=cur,
        current_price=round(price, 4), pnl_pct=round(pnl, 2),
        market_value=round(price * shares, 2),
        source=q.source,
    )


def _krw_equivalent(total_krw: float, total_usd: float,
                    live_rate: float | None, assumed_rate: float | None
                    ) -> tuple[float | None, float | None, str]:
    """USD 평가액을 환율로 KRW 환산 후 합산. 실시간(live) 우선, 없으면 스냅샷 가정환율.

    반환: (krw_equiv, rate_used, source). source ∈ {"live", "assumed", "none"}.
    실시간 값이 0/음수면 신뢰 불가로 보고 가정환율로 폴백.
    """
    if live_rate and live_rate > 0:
        rate, source = live_rate, "live"
    elif assumed_rate:
        rate, source = assumed_rate, "assumed"
    else:
        return None, None, "none"
    return round(total_krw + total_usd * rate, 2), round(rate, 2), source


def fetch_portfolio(usd_krw_rate: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

    # 고정 가정환율(스냅샷) 대신 실시간 FX로 USD→KRW 환산해 합산 총액 제공
    assumed_rate = portfolio_data.get("totals", {}).get("fxAssumedUSDKRW")
    krw_equiv, rate_used, rate_source = _krw_equivalent(
        total_krw, total_usd, usd_krw_rate, assumed_rate)

    summary = {
        "total_value_krw": round(total_krw, 2),
        "total_value_usd": round(total_usd, 2),
        "total_value_krw_equiv": krw_equiv,
        "fx_rate_used": rate_used,
        "fx_rate_source": rate_source,
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
    live_usd_krw = snapshot.fx.get("usd_krw", {}).get("price")
    snapshot.portfolio, snapshot.portfolio_summary = fetch_portfolio(usd_krw_rate=live_usd_krw)
    log.info("Fetching watchlist...")
    snapshot.watchlist = fetch_watchlist(load_watchlist())
    log.info("Fetching monitored universe...")
    snapshot.universe = fetch_universe()

    snapshot_path = SNAPSHOTS_DIR / f"{now_kst.strftime('%Y%m%d-%H%M')}.json"
    snapshot_dict = asdict(snapshot)
    snapshot_path.write_text(json.dumps(snapshot_dict, indent=2, default=str, ensure_ascii=False))
    LATEST_FILE.write_text(json.dumps(snapshot_dict, indent=2, default=str, ensure_ascii=False))

    try:
        from corvin_jarvis import timeseries
        rows_written = timeseries.write_snapshot(TIMESERIES_DB, snapshot_dict)
        log.info("timeseries: %d rows persisted", rows_written)
    except Exception as e:  # noqa: BLE001
        log.warning("timeseries write failed (non-fatal): %s", e)

    log.info("Pulse 완료 — %s", snapshot_path.name)
    return snapshot_path


def _print_summary(snapshot_path: Path) -> None:
    data = json.loads(snapshot_path.read_text())
    log.info("=== Pulse Summary ===")
    for name, q in data["indices"].items():
        log.info("  [IDX] %-8s %s  (%+.2f%%) %s",
                 name, q.get("price"), q.get("pct_change") or 0, q.get("source", ""))
    for name, q in data["commodities"].items():
        log.info("  [CMD] %-8s %s  (%+.2f%%)", name, q.get("price"), q.get("pct_change") or 0)
    for name, q in data["fx"].items():
        log.info("  [FX]  %-8s %s  (%+.2f%%)", name, q.get("price"), q.get("pct_change") or 0)
    log.info("  [PFL] positions=%s value_KRW=%s value_USD=%s",
             data["portfolio_summary"].get("position_count"),
             data["portfolio_summary"].get("total_value_krw"),
             data["portfolio_summary"].get("total_value_usd"))
    for w in data.get("watchlist", []):
        if w.get("price") is not None:
            log.info("  [WL]  %-8s %s  (%+.2f%%) %s",
                     w["symbol"], w["price"], w.get("pct_change") or 0, w.get("source", ""))


if __name__ == "__main__":
    path = run_pulse()
    _print_summary(path)
