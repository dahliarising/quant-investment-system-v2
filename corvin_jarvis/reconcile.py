"""Corvin Jarvis — Portfolio Reconciliation (Phase 8)

portfolio.json을 안전하게 업데이트. 매수/매도 액션을 trade log에 기록.
실제 주문 실행 X — advisory + bookkeeping only (CLAUDE.md 규칙).

사용:
    # 종목 추가
    python corvin_jarvis/reconcile.py add META 3 620.50 USD
    # 종목 제거 / 매도
    python corvin_jarvis/reconcile.py sell 005930.KS 22 270500 KRW
    # 부분 매도
    python corvin_jarvis/reconcile.py sell 000660.KS 2 1819000 KRW
    # 단순 PnL 업데이트 (현재가만 재조회)
    python corvin_jarvis/reconcile.py refresh
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PORTFOLIO_FILE = PROJECT_ROOT / "portfolio.json"
TRADES_FILE = BASE_DIR / "state" / "trades.json"
BACKUP_DIR = BASE_DIR / "state" / "portfolio_backups"
LOG_FILE = BASE_DIR / "state" / "reconcile.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("corvin.reconcile")


@dataclass(frozen=True)
class Trade:
    action: str  # "buy" | "sell"
    symbol: str
    shares: float
    price: float
    currency: str
    timestamp_iso: str
    note: str = ""


def _load_portfolio() -> dict[str, Any]:
    if not PORTFOLIO_FILE.exists():
        return {"holdings": [], "updatedAt": date.today().isoformat()}
    return json.loads(PORTFOLIO_FILE.read_text())


def _backup_portfolio() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy(PORTFOLIO_FILE, BACKUP_DIR / f"portfolio-{ts}.json")


def _save_portfolio(portfolio: dict[str, Any]) -> None:
    portfolio["updatedAt"] = date.today().isoformat()
    _backup_portfolio()
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2, ensure_ascii=False))


def _append_trade(trade: Trade) -> None:
    if TRADES_FILE.exists():
        log_data = json.loads(TRADES_FILE.read_text())
    else:
        log_data = {"trades": []}
    log_data["trades"].append({
        "action": trade.action, "symbol": trade.symbol, "shares": trade.shares,
        "price": trade.price, "currency": trade.currency,
        "timestamp": trade.timestamp_iso, "note": trade.note,
    })
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRADES_FILE.write_text(json.dumps(log_data, indent=2, ensure_ascii=False))


def buy(symbol: str, shares: float, price: float, currency: str, note: str = "") -> None:
    portfolio = _load_portfolio()
    holdings = portfolio["holdings"]
    existing = next((h for h in holdings if h["symbol"] == symbol), None)

    if existing:
        old_shares = existing["shares"]
        old_avg = existing["avgPrice"]
        new_total = old_shares + shares
        new_avg = (old_shares * old_avg + shares * price) / new_total if new_total else 0
        existing["shares"] = new_total
        existing["avgPrice"] = round(new_avg, 4)
        log.info("%s: %s주 추가매수 @ %s%s → 총 %s주 @ %s 평단",
                 symbol, shares, currency, price, new_total, round(new_avg, 4))
    else:
        holdings.append({
            "symbol": symbol, "shares": shares, "avgPrice": price,
            "currency": currency, "addedAt": date.today().isoformat(),
        })
        log.info("%s: 신규 진입 %s주 @ %s%s", symbol, shares, currency, price)

    _save_portfolio(portfolio)
    _append_trade(Trade(
        action="buy", symbol=symbol, shares=shares, price=price,
        currency=currency, timestamp_iso=datetime.now(timezone.utc).isoformat(), note=note,
    ))


def sell(symbol: str, shares: float, price: float, currency: str, note: str = "") -> None:
    portfolio = _load_portfolio()
    holdings = portfolio["holdings"]
    existing = next((h for h in holdings if h["symbol"] == symbol), None)
    if existing is None:
        log.error("%s 보유 중 아님 — sell 무시", symbol)
        return

    if shares >= existing["shares"]:
        realized_pnl_pct = (price / existing["avgPrice"] - 1) * 100
        portfolio["holdings"] = [h for h in holdings if h["symbol"] != symbol]
        log.info("%s 전량 매도 — 실현 PnL %+.2f%% (%s주 @ %s%s)",
                 symbol, realized_pnl_pct, existing["shares"], currency, price)
    else:
        existing["shares"] -= shares
        realized_pnl_pct = (price / existing["avgPrice"] - 1) * 100
        log.info("%s 부분 매도 %s주 — 잔여 %s주 / 실현 PnL %+.2f%%",
                 symbol, shares, existing["shares"], realized_pnl_pct)

    _save_portfolio(portfolio)
    _append_trade(Trade(
        action="sell", symbol=symbol, shares=shares, price=price,
        currency=currency, timestamp_iso=datetime.now(timezone.utc).isoformat(),
        note=f"{note} | realized_pnl_pct={realized_pnl_pct:.2f}",
    ))


def refresh() -> None:
    """pulse.py 호출해서 portfolio 평가만 갱신 (보유 변경 없음)."""
    sys.path.insert(0, str(BASE_DIR))
    from pulse import run_pulse  # noqa: PLC0415
    run_pulse()
    log.info("Refresh 완료 — state/latest.json 업데이트됨")


def _print_usage() -> None:
    print(__doc__)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        _print_usage()
        return
    cmd = argv[1]
    if cmd == "refresh":
        refresh()
    elif cmd == "buy" and len(argv) >= 6:
        buy(argv[2], float(argv[3]), float(argv[4]), argv[5], note=" ".join(argv[6:]))
    elif cmd in ("sell", "remove") and len(argv) >= 6:
        sell(argv[2], float(argv[3]), float(argv[4]), argv[5], note=" ".join(argv[6:]))
    else:
        _print_usage()


if __name__ == "__main__":
    main(sys.argv)
