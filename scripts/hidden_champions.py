#!/usr/bin/env python3
"""히든 챔피언 발굴 — 저PER + 높은 성장."""
import json, sys, argparse
import yfinance as yf

CANDIDATES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AMD", "INTC", "CRM", "ADBE", "NFLX", "SHOP", "SQ", "PLTR",
    "ABNB", "UBER", "SNAP", "PINS", "DDOG", "ZS", "CRWD", "NET",
    "005930.KS", "000660.KS", "035420.KS", "035720.KS", "068270.KS",
    "028260.KS", "034730.KS", "003670.KS", "006800.KS", "009150.KS",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default=None)
    args = parser.parse_args()

    pool = CANDIDATES
    if args.market == "us":
        pool = [t for t in pool if not t.endswith(".KS")]
    elif args.market == "kr":
        pool = [t for t in pool if t.endswith(".KS")]

    results = []
    for sym in pool:
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            per = info.get("trailingPE") or info.get("forwardPE")
            growth = info.get("revenueGrowth")
            if per and per < 25 and growth and growth > 0.1:
                results.append({
                    "종목": sym,
                    "이름": info.get("shortName", "N/A"),
                    "PER": round(per, 2),
                    "매출성장률(%)": round(growth * 100, 1),
                    "시가총액": info.get("marketCap", 0),
                })
        except Exception:
            continue

    results.sort(key=lambda x: x.get("매출성장률(%)", 0), reverse=True)
    print(json.dumps(results[:10], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
