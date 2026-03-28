#!/usr/bin/env python3
"""포트폴리오 최적화 — 마코위츠 기반 최적 비중."""
import json, sys, argparse
import numpy as np
import yfinance as yf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="쉼표 구분 종목코드")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if len(tickers) < 2:
        print(json.dumps({"에러": "최소 2개 종목이 필요합니다."}))
        sys.exit(1)

    try:
        data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)["Close"]
        if data.empty:
            print(json.dumps({"에러": "데이터를 가져올 수 없습니다."}))
            sys.exit(1)

        # 데이터 없는 종목 제거 (부분 다운로드 가드)
        if hasattr(data, 'columns'):
            valid_cols = data.dropna(axis=1, how='all').columns
            dropped = set(tickers) - set(valid_cols)
            if dropped:
                sys.stderr.write(f"[WARN] 데이터 없는 종목 제외: {dropped}\n")
            data = data[valid_cols]
            tickers = list(valid_cols)
        if len(tickers) < 2:
            print(json.dumps({"에러": "유효한 데이터가 2개 종목 미만입니다."}))
            sys.exit(1)

        returns = data.pct_change().dropna()
        mean_ret = returns.mean() * 252
        cov = returns.cov() * 252

        # 랜덤 포트폴리오 시뮬레이션 (몬테카를로)
        n = len(tickers)
        best_sharpe = -999
        best_weights = None
        best_ret = 0
        best_vol = 0

        np.random.seed(42)
        for _ in range(5000):
            w = np.random.random(n)
            w /= w.sum()
            port_ret = np.dot(w, mean_ret)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov.values, w)))
            sharpe = port_ret / port_vol if port_vol > 0 else 0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w
                best_ret = port_ret
                best_vol = port_vol

        result = {
            "최적_포트폴리오": {
                t: f"{round(w * 100, 1)}%" for t, w in zip(tickers, best_weights)
            },
            "기대수익률(%)": round(best_ret * 100, 2),
            "변동성(%)": round(best_vol * 100, 2),
            "샤프비율": round(best_sharpe, 3),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"에러": f"최적화 실패: {e}"}, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
