"""1회 실행 — 지수·매크로 + 감시종목 일봉 ~10년 초기 적재."""
from pathlib import Path
import json

from corvin_jarvis.prediction import backfill

_DB = Path(__file__).resolve().parent.parent / "state" / "daily_history.db"
_FEATURES = [("kospi", "KR"), ("kosdaq", "KR"), ("nasdaq", "US"), ("sp500", "US"),
             ("vix", "US"), ("usd_krw", "KR"), ("gold", "US"), ("copper", "US"),
             ("dxy", "US")]

# FDR 심볼 매핑 (내부키 → fdr 심볼) — 2026-06-13 실측 검증 완료.
# gold: ZG=F는 404 → GC=F로 교정. 나머지는 모두 가용 확인.
_FDR_SYMBOL = {"kospi": "KS11", "kosdaq": "KQ11", "nasdaq": "IXIC", "sp500": "US500",
               "vix": "VIX", "usd_krw": "USD/KRW", "gold": "GC=F", "copper": "HG=F",
               "dxy": "DX-Y.NYB"}


def _num(val, default=0.0):
    """NaN/None-safe float. 지수·FX·선물은 Volume이 NaN인 행이 있어 int(NaN) 폭발 방지."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return default if f != f else f  # f != f → NaN


def _fetch(symbol, market, start):
    import FinanceDataReader as fdr
    fsym = _FDR_SYMBOL.get(symbol, symbol)
    df = fdr.DataReader(fsym, start)
    rows = []
    for idx, row in df.iterrows():
        rows.append({"symbol": symbol, "date": idx.strftime("%Y-%m-%d"),
                     "open": _num(row.get("Open", 0)), "high": _num(row.get("High", 0)),
                     "low": _num(row.get("Low", 0)), "close": _num(row.get("Close", 0)),
                     "volume": int(_num(row.get("Volume", 0))), "source": "fdr"})
    return rows


def main():
    backfill.init_db(_DB)
    uni = json.loads((Path(__file__).resolve().parent.parent / "monitored_universe.json").read_text())
    syms = list(_FEATURES) + [(t["symbol"], t.get("market", "KR")) for t in uni.get("tickers", [])]
    n = backfill.incremental_update(_DB, syms, fetcher=_fetch)
    print(f"backfill 완료: {n} rows, db={_DB}")


if __name__ == "__main__":
    main()
