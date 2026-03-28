#!/usr/bin/env python3
"""일일 브리핑 — 주요 지수 + 주목할 종목."""
import json, sys
import yfinance as yf

INDICES = {"S&P 500": "^GSPC", "나스닥": "^IXIC", "코스피": "^KS11", "USD/KRW": "KRW=X"}
WATCHLIST = ["AAPL", "NVDA", "TSLA", "005930.KS", "000660.KS"]

def main():
    briefing = {"지수": {}, "주목_종목": []}

    # 지수
    for name, sym in INDICES.items():
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            price = getattr(fi, "last_price", None)
            prev = getattr(fi, "previous_close", None)
            chg = round((price - prev) / prev * 100, 2) if price and prev else None
            briefing["지수"][name] = {"가격": round(price, 2) if price else "N/A", "등락(%)": chg}
        except Exception:
            briefing["지수"][name] = "조회 실패"

    # 워치리스트
    for sym in WATCHLIST:
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            info = t.info or {}
            price = getattr(fi, "last_price", None)
            prev = getattr(fi, "previous_close", None)
            chg = round((price - prev) / prev * 100, 2) if price and prev else None
            briefing["주목_종목"].append({
                "종목": sym,
                "이름": info.get("shortName", "N/A"),
                "현재가": round(price, 2) if price else "N/A",
                "등락(%)": chg,
            })
        except Exception:
            continue

    print(json.dumps(briefing, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
