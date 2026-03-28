#!/usr/bin/env python3
"""종목 스크리닝 — 조건 기반 필터."""
import json, sys, argparse
import yfinance as yf

# 대표 종목 풀 (확장 가능)
US_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "ADBE",
    "CRM", "NFLX", "AMD", "INTC", "PFE", "ABBV", "KO", "PEP", "MRK",
    "TMO", "AVGO", "CSCO", "ACN", "MCD", "WMT", "COST", "LLY",
]
KR_TICKERS = [
    "005930.KS", "000660.KS", "035420.KS", "035720.KS", "005380.KS",
    "051910.KS", "006400.KS", "068270.KS", "003550.KS", "105560.KS",
    "028260.KS", "012330.KS", "066570.KS", "055550.KS", "034730.KS",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="all", help="us, kr, or all")
    parser.add_argument("--sector", default=None)
    parser.add_argument("--minMarketCap", type=float, default=0)
    parser.add_argument("--maxPER", type=float, default=9999)
    args = parser.parse_args()

    pool = []
    if args.market in ("us", "all"):
        pool += US_TICKERS
    if args.market in ("kr", "all"):
        pool += KR_TICKERS

    results = []
    if len(pool) > 30:
        sys.stderr.write(f"[INFO] 스크리닝 대상: {len(pool)}개 종목 (시간이 걸릴 수 있습니다)\n")
    for sym in pool:  # 전체 풀 처리
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            mcap = info.get("marketCap", 0) or 0
            per = info.get("trailingPE") or info.get("forwardPE") or 9999
            sector = info.get("sector", "")

            if mcap < args.minMarketCap:
                continue
            if per > args.maxPER:
                continue
            if args.sector and args.sector.lower() not in sector.lower():
                continue

            results.append({
                "종목": sym,
                "이름": info.get("shortName", "N/A"),
                "시가총액": mcap,
                "PER": round(per, 2) if per < 9999 else "N/A",
                "섹터": sector,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x.get("시가총액", 0), reverse=True)
    print(json.dumps(results[:20], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
