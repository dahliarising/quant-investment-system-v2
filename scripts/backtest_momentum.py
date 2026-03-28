#!/usr/bin/env python3
"""모멘텀 백테스트 — 단순 모멘텀 전략 시뮬레이션."""
import json, sys, argparse
import numpy as np
import yfinance as yf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--strategy", default="momentum")
    args = parser.parse_args()

    try:
        data = yf.download(args.ticker, period=args.period, auto_adjust=True, progress=False)
        if data.empty:
            print(json.dumps({"에러": f"{args.ticker} 데이터를 가져올 수 없습니다."}))
            sys.exit(1)

        close = data["Close"].values.flatten()
        returns = np.diff(close) / close[:-1]

        # 단순 모멘텀: 20일 수익률 양수면 매수
        lookback = 20
        signals = []
        for i in range(lookback, len(close)):
            mom = (close[i] - close[i - lookback]) / close[i - lookback]
            signals.append(1 if mom > 0 else 0)

        # 시그널은 오늘 종가 기반 → 내일 수익률에 적용 (룩어헤드 바이어스 방지)
        # signals[i]는 close[lookback+i] 시점 시그널 → returns[lookback+i]가 아닌 returns[lookback+i+1]에 적용
        next_day_returns = returns[lookback + 1:]
        aligned_signals = np.array(signals[:len(next_day_returns)])
        strategy_returns = next_day_returns * aligned_signals

        cum_strategy = float(np.prod(1 + strategy_returns) - 1)
        cum_buyhold = float(np.prod(1 + returns) - 1)

        # 최대낙폭: cumprod 기반 equity curve → running peak 대비 하락폭
        equity = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(equity)
        max_drawdown = float(np.min(equity - running_max))

        result = {
            "종목": args.ticker,
            "기간": args.period,
            "전략": args.strategy,
            "전략_수익률(%)": round(cum_strategy * 100, 2),
            "바이앤홀드_수익률(%)": round(cum_buyhold * 100, 2),
            "최대낙폭(%)": round(max_drawdown * 100, 2),
            "거래일수": len(close),
            "매수_신호_비율(%)": round(sum(signals) / len(signals) * 100, 1) if signals else 0,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"에러": f"백테스트 실패: {e}"}, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
