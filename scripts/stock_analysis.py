#!/usr/bin/env python3
"""개별 종목 분석 — 재무 지표 + 기술적 데이터."""
import json, sys
import yfinance as yf

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"에러": "종목 코드를 입력하세요. 예: python3 stock_analysis.py AAPL"}))
        sys.exit(1)

    ticker = sys.argv[1]
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        fast = t.fast_info

        result = {
            "종목": ticker,
            "이름": info.get("longName") or info.get("shortName", "N/A"),
            "현재가": getattr(fast, "last_price", None),
            "시가총액": info.get("marketCap"),
            "PER": info.get("trailingPE") or info.get("forwardPE"),
            "PBR": info.get("priceToBook"),
            "배당률(%)": info.get("dividendYield"),
            "52주_최고": info.get("fiftyTwoWeekHigh"),
            "52주_최저": info.get("fiftyTwoWeekLow"),
            "섹터": info.get("sector", "N/A"),
            "산업": info.get("industry", "N/A"),
            "매출": info.get("totalRevenue"),
            "영업이익": info.get("operatingIncome"),
            "순이익": info.get("netIncomeToCommon"),
        }

        # 수치 포매팅
        for k, v in result.items():
            if isinstance(v, float):
                result[k] = round(v, 2)

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"에러": f"{ticker} 분석 실패: {e}"}, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
