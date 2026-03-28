#!/usr/bin/env python3
"""시장 개요 — 미국/한국 주요 지수 + 환율."""
import json, sys
import yfinance as yf

INDICES = {
    "S&P 500": "^GSPC",
    "나스닥": "^IXIC",
    "다우존스": "^DJI",
    "코스피": "^KS11",
    "코스닥": "^KQ11",
    "USD/KRW": "KRW=X",
}

def main():
    result = {}
    for name, symbol in INDICES.items():
        try:
            t = yf.Ticker(symbol)
            info = t.fast_info
            price = getattr(info, "last_price", None)
            prev = getattr(info, "previous_close", None)
            if price and prev:
                change_pct = round((price - prev) / prev * 100, 2)
            else:
                change_pct = None
            result[name] = {
                "현재가": round(price, 2) if price else "N/A",
                "전일대비(%)": change_pct if change_pct is not None else "N/A",
            }
        except Exception as e:
            result[name] = {"에러": str(e)}
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
