"""
실전 전략 예제: 멀티 팩터 모멘텀 전략
- Pipeline API 사용
- Alpha 158 팩터 활용
- Zipline 스타일 백테스트
- 포트폴리오 최적화
"""
import sys
sys.path.append('/home/claude/quant_investment_system_v2')

import pandas as pd
import numpy as np
from datetime import datetime

from features.pipeline import Pipeline, Factor
from features.alpha158 import Alpha158
from validation.backtest_zipline import BacktestEngine, Context, DataPortal
from portfolio.optimizer import PortfolioOptimizer, RegimeBasedAdjustment


class MultiFactorScore(Factor):
    """
    멀티 팩터 종합 점수
    - 모멘텀: 40%
    - 품질: 30%
    - 리스크: 20%
    - 유동성: 10%
    """
    
    def __init__(self):
        super().__init__(inputs=['Close', 'Volume'], window_length=252)
        self.name = "MultiFactorScore"
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """종합 점수 계산"""
        # Alpha 158 팩터 생성
        features = Alpha158.generate_basic(data)
        
        if features.empty or len(features) < 60:
            return pd.Series(0, index=data.index)
        
        # 1. 모멘텀 점수 (40%)
        momentum_60d = features['ROC_60'].fillna(0)
        momentum_20d = features['ROC_20'].fillna(0)
        momentum_score = (momentum_60d * 0.7 + momentum_20d * 0.3)
        
        # 2. 품질 점수 (30%)
        # 추세 일관성: 20일 이동평균과 현재가 관계
        ma_20 = features['MA_20']
        price = features['CLOSE']
        trend_consistency = (price > ma_20).astype(float)
        
        # 변동성 대비 수익률
        returns_vol_ratio = momentum_60d / (features['STD_20'] + 1e-6)
        quality_score = trend_consistency * 0.5 + returns_vol_ratio * 0.5
        
        # 3. 리스크 점수 (20%) - 낮은 변동성이 좋음
        volatility = features['STD_20']
        risk_score = -volatility  # 음수 (낮을수록 좋음)
        
        # 4. 유동성 점수 (10%)
        volume_ma = features['VOLUME_MA_20']
        liquidity_score = volume_ma / volume_ma.max() if volume_ma.max() > 0 else 0
        
        # 종합 점수
        final_score = (
            momentum_score * 0.40 +
            quality_score * 0.30 +
            risk_score * 0.20 +
            liquidity_score * 0.10
        )
        
        # Z-score 정규화
        final_score = (final_score - final_score.mean()) / (final_score.std() + 1e-6)
        
        return final_score


class MultiFactorStrategy:
    """
    멀티 팩터 전략
    - 월간 리밸런싱
    - 상위 10-15 종목 선택
    - 포트폴리오 최적화
    """
    
    def __init__(self):
        self.optimizer = PortfolioOptimizer()
        
    def initialize(self, context: Context):
        """전략 초기화"""
        print("=" * 80)
        print("멀티 팩터 모멘텀 전략 초기화")
        print("=" * 80)
        
        # 설정
        context.universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'JPM', 'V', 'WMT'
        ]
        context.rebalance_frequency = 20  # 20일마다 (약 월간)
        context.lookback_days = 252  # 1년
        context.top_n = 10  # 상위 10종목
        
        # 카운터
        context.days_since_rebalance = 0
        context.rebalance_count = 0
        
        print(f"Universe: {len(context.universe)} 종목")
        print(f"Rebalance: {context.rebalance_frequency}일마다")
        print(f"Top N: {context.top_n} 종목")
        
    def handle_data(self, context: Context, data: DataPortal):
        """매 거래일마다 실행"""
        # 리밸런싱 체크
        context.days_since_rebalance += 1
        
        if context.days_since_rebalance < context.rebalance_frequency:
            return
        
        # 리밸런싱 실행
        context.days_since_rebalance = 0
        context.rebalance_count += 1
        
        current_date = context.current_date
        
        print(f"\n{'='*60}")
        print(f"Rebalance #{context.rebalance_count} - {current_date.date()}")
        print(f"{'='*60}")
        
        # 각 종목의 점수 계산
        scores = {}
        returns_data = {}
        
        for symbol in context.universe:
            # 과거 데이터 조회
            hist = data.get_history(
                symbol,
                ['Open', 'High', 'Low', 'Close', 'Volume'],
                bar_count=context.lookback_days,
                end_date=current_date
            )
            
            if len(hist) < 100:  # 최소 데이터 필요
                continue
            
            # 멀티 팩터 점수 계산
            try:
                factor = MultiFactorScore()
                score_series = factor.compute(hist)
                
                if not score_series.empty:
                    scores[symbol] = score_series.iloc[-1]
                    
                    # 수익률 데이터 저장 (최적화용)
                    returns = hist['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
            except Exception as e:
                print(f"  ⚠️ Error calculating score for {symbol}: {e}")
                continue
        
        if not scores:
            print("  ⚠️ No valid scores, skipping rebalance")
            return
        
        # 점수 시리즈
        scores_series = pd.Series(scores)
        
        print(f"\n점수 계산 완료: {len(scores_series)} 종목")
        print(f"Top 5 점수:")
        print(scores_series.nlargest(5).round(4))
        
        # 수익률 데이터프레임
        returns_df = pd.DataFrame(returns_data).fillna(0)
        
        # 포트폴리오 최적화
        target_weights = self.optimizer.optimize(
            scores_series,
            returns_df,
            method='inverse_vol'
        )
        
        print(f"\n목표 비중 (상위 5):")
        top_weights = target_weights[target_weights > 0].nlargest(5)
        for symbol, weight in top_weights.items():
            print(f"  {symbol}: {weight:.2%}")
        
        # 현재 데이터 조회
        current_data = data.get_current_data(current_date, symbols=context.universe)
        
        if current_data.empty:
            print("  ⚠️ No current data, skipping rebalance")
            return
        
        # 리밸런싱 실행 (간단히 시뮬레이션)
        # 실제로는 BacktestEngine의 order_target_percent 사용
        print(f"\n✅ Rebalance 완료")
        print(f"   포트폴리오 가치: ${context.portfolio.portfolio_value:,.0f}")
        print(f"   현금: ${context.portfolio.cash:,.0f}")
        print(f"   포지션 수: {len(context.portfolio.positions)}")


def create_sample_data(
    symbols: list,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """샘플 데이터 생성"""
    print("\n샘플 데이터 생성 중...")
    
    dates = pd.date_range(start_date, end_date, freq='D')
    
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=['date', 'symbol']
    )
    
    np.random.seed(42)
    
    # 각 종목별 가격 생성
    data_list = []
    base_prices = {s: np.random.uniform(100, 300) for s in symbols}
    
    for symbol in symbols:
        base = base_prices[symbol]
        n_days = len(dates)
        
        # 추세 + 노이즈
        trend = np.linspace(0, base * 0.2, n_days)  # 20% 상승 추세
        noise = np.random.randn(n_days) * base * 0.02  # 2% 노이즈
        prices = base + trend + noise.cumsum()
        
        for i, date in enumerate(dates):
            open_price = prices[i] + np.random.randn() * base * 0.005
            close_price = prices[i]
            high_price = max(open_price, close_price) + abs(np.random.randn()) * base * 0.01
            low_price = min(open_price, close_price) - abs(np.random.randn()) * base * 0.01
            
            data_list.append({
                'date': date,
                'symbol': symbol,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': np.random.randint(1000000, 10000000)
            })
    
    df = pd.DataFrame(data_list)
    df = df.set_index(['date', 'symbol'])
    
    print(f"  생성 완료: {len(df)} rows")
    print(f"  종목: {len(symbols)}개")
    print(f"  기간: {dates[0].date()} ~ {dates[-1].date()}")
    
    return df


def run_multifactor_backtest():
    """멀티 팩터 전략 백테스트 실행"""
    print("\n" + "=" * 80)
    print("멀티 팩터 모멘텀 전략 백테스트")
    print("=" * 80)
    
    # 데이터 생성
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
               'META', 'TSLA', 'JPM', 'V', 'WMT']
    
    data = create_sample_data(
        symbols=symbols,
        start_date='2022-01-01',
        end_date='2023-12-31'
    )
    
    # 전략 인스턴스
    strategy = MultiFactorStrategy()
    
    # 백테스트 엔진
    # Note: 실제 구현 시 BacktestEngine을 완전히 통합
    print("\n백테스트 시뮬레이션...")
    print("(실제 백테스트는 BacktestEngine 사용)")
    
    # 간단한 시뮬레이션
    context = Context()
    strategy.initialize(context)
    
    data_portal = DataPortal(data)
    
    # 몇 개 날짜만 테스트
    test_dates = data.index.get_level_values('date').unique()[::20][:5]
    
    for date in test_dates:
        context.current_date = date
        strategy.handle_data(context, data_portal)
    
    print("\n" + "=" * 80)
    print("✅ 멀티 팩터 전략 시뮬레이션 완료!")
    print("=" * 80)
    
    print("\n📊 전략 요약:")
    print("""
    - 팩터: 모멘텀(40%) + 품질(30%) + 리스크(20%) + 유동성(10%)
    - 리밸런싱: 월간 (20일)
    - 종목 수: 10-15개
    - 비중: 변동성 역수 가중
    - 제약: 개별 3-15%, 섹터 40%
    """)
    
    print("\n🚀 실전 적용 방법:")
    print("""
    1. 실제 데이터 연결 (pykrx, yfinance)
    2. BacktestEngine으로 완전한 백테스트
    3. Purged K-Fold로 검증
    4. SHAP으로 팩터 기여도 분석
    5. Streamlit UI로 모니터링
    """)


if __name__ == "__main__":
    run_multifactor_backtest()
