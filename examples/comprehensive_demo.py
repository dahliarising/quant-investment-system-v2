"""
퀀트 투자 시스템 종합 예제
- 이벤트 엔진
- Gateway 패턴
- Pipeline API
- Alpha 158
- Zipline 스타일 백테스트
"""
import sys
sys.path.append('/home/claude/quant_investment_system_v2')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.event_engine import EventEngine, EventType, Event
from engine.main_engine import MainEngine
from data.gateway import KRDataGateway, USDataGateway
from features.pipeline import Pipeline, FactorLibrary
from features.alpha158 import Alpha158


def example_1_event_system():
    """예제 1: 이벤트 시스템"""
    print("\n" + "=" * 80)
    print("예제 1: 이벤트 기반 시스템 (VN.py 스타일)")
    print("=" * 80)
    
    # 이벤트 엔진 생성
    engine = EventEngine()
    
    # 핸들러 정의
    def on_market_data(event: Event):
        print(f"  📊 Market Data: {event.data}")
    
    def on_signal(event: Event):
        print(f"  🎯 Signal: {event.data}")
    
    def on_portfolio_update(event: Event):
        print(f"  💼 Portfolio: {event.data}")
    
    # 핸들러 등록
    engine.register(EventType.MARKET_DATA, on_market_data)
    engine.register(EventType.SIGNAL_GENERATED, on_signal)
    engine.register(EventType.PORTFOLIO_REBALANCE, on_portfolio_update)
    
    # 엔진 시작
    engine.start()
    
    # 이벤트 발행 시뮬레이션
    print("\n이벤트 발행 중...")
    engine.put(Event(EventType.MARKET_DATA, {"symbol": "005930", "price": 70000}))
    engine.put(Event(EventType.SIGNAL_GENERATED, {"symbol": "005930", "action": "BUY"}))
    engine.put(Event(EventType.PORTFOLIO_REBALANCE, {"portfolio_value": 105_000_000}))
    
    import time
    time.sleep(1)
    
    # 엔진 정지
    engine.stop()
    print("\n✅ 이벤트 시스템 예제 완료")


def example_2_main_engine():
    """예제 2: 메인 엔진 및 Gateway"""
    print("\n" + "=" * 80)
    print("예제 2: 메인 엔진 및 Data Gateway (VN.py 스타일)")
    print("=" * 80)
    
    # 메인 엔진 생성
    main_engine = MainEngine()
    
    # 데이터 수신 핸들러
    def on_data(event: Event):
        bars = event.data.get('bars')
        if bars is not None and not bars.empty:
            print(f"\n  📈 데이터 수신: {bars.shape[0]}개 봉")
            print(f"     기간: {bars.index.min()} ~ {bars.index.max()}")
            print(f"     종가 범위: {bars['Close'].min():.0f} ~ {bars['Close'].max():.0f}")
    
    main_engine.event_engine.register(EventType.HISTORICAL_DATA, on_data)
    
    # 엔진 시작
    main_engine.start()
    
    # Gateway 추가
    kr_gateway = main_engine.add_gateway(KRDataGateway)
    
    if main_engine.connect_gateway('KRDataGateway'):
        # 데이터 조회
        print("\n삼성전자 데이터 조회 중...")
        df = kr_gateway.get_bars('005930', '2024-01-01', '2024-02-01')
    
    import time
    time.sleep(1)
    
    # 엔진 정지
    main_engine.stop()
    print("\n✅ 메인 엔진 예제 완료")


def example_3_pipeline():
    """예제 3: Pipeline API"""
    print("\n" + "=" * 80)
    print("예제 3: Pipeline API를 이용한 팩터 계산 (Zipline 스타일)")
    print("=" * 80)
    
    # 샘플 데이터 생성
    print("\n샘플 데이터 생성 중...")
    dates = pd.date_range('2023-06-01', '2024-01-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=['date', 'symbol']
    )
    
    np.random.seed(42)
    base_prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 120}
    
    data_list = []
    for symbol in symbols:
        base = base_prices[symbol]
        n_days = len(dates)
        prices = base + np.cumsum(np.random.randn(n_days) * 2)
        
        for i, date in enumerate(dates):
            data_list.append({
                'date': date,
                'symbol': symbol,
                'Open': prices[i] + np.random.rand() - 0.5,
                'High': prices[i] + abs(np.random.rand()),
                'Low': prices[i] - abs(np.random.rand()),
                'Close': prices[i],
                'Volume': np.random.randint(1000000, 10000000)
            })
    
    data = pd.DataFrame(data_list)
    data = data.set_index(['date', 'symbol'])
    
    print(f"  데이터 Shape: {data.shape}")
    print(f"  기간: {dates.min().date()} ~ {dates.max().date()}")
    
    # Pipeline 생성
    print("\nPipeline 생성 중...")
    pipeline = FactorLibrary.create_default_pipeline(universe=['AAPL', 'MSFT'])
    
    # 실행
    print("Pipeline 실행 중...")
    factors = pipeline.run(data, start_date='2024-01-01')
    
    print(f"\n  ✅ 계산된 팩터: {len(factors.columns)}개")
    print(f"  팩터 목록 (일부): {factors.columns.tolist()[:10]}")
    
    # AAPL 팩터 확인
    print("\n📊 AAPL 최근 팩터 값:")
    aapl_factors = factors.xs('AAPL', level='symbol').tail()
    print(aapl_factors[['Returns_20d', 'Returns_60d', 'Volatility_20d']].round(4))
    
    print("\n✅ Pipeline 예제 완료")


def example_4_alpha158():
    """예제 4: Alpha 158 팩터"""
    print("\n" + "=" * 80)
    print("예제 4: Alpha 158 팩터 세트 (Qlib 스타일)")
    print("=" * 80)
    
    # 샘플 데이터
    print("\n샘플 데이터 생성 중...")
    dates = pd.date_range('2023-01-01', '2024-01-31', freq='D')
    
    np.random.seed(42)
    df = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 1.5),
        'High': 102 + np.cumsum(np.random.randn(len(dates)) * 1.5),
        'Low': 98 + np.cumsum(np.random.randn(len(dates)) * 1.5),
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 1.5),
        'Volume': np.random.randint(5000000, 15000000, len(dates))
    }, index=dates)
    
    # High/Low 조정
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(len(dates)) * 0.5)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(len(dates)) * 0.5)
    
    print(f"  데이터 Shape: {df.shape}")
    
    # 기본 팩터 생성
    print("\n기본 팩터 생성 중...")
    basic_features = Alpha158.generate_basic(df)
    
    print(f"  ✅ 기본 팩터: {basic_features.shape[1]}개")
    print(f"  컬럼 (일부): {basic_features.columns.tolist()[:8]}")
    
    # 전체 팩터 생성
    print("\n전체 Alpha158 팩터 생성 중...")
    all_features = Alpha158.generate_all(df)
    
    print(f"  ✅ 전체 팩터: {all_features.shape[1]}개")
    
    # 최근 팩터 값
    print("\n📊 최근 팩터 값 (일부):")
    sample_cols = ['CLOSE', 'ROC_5', 'ROC_20', 'MA_20', 'STD_20', 'VOLUME_MA_20']
    print(all_features[sample_cols].tail().round(2))
    
    print("\n✅ Alpha 158 예제 완료")


def example_5_integrated():
    """예제 5: 통합 워크플로우"""
    print("\n" + "=" * 80)
    print("예제 5: 통합 워크플로우 - 전체 시스템 연동")
    print("=" * 80)
    
    # 1. 메인 엔진 시작
    print("\n1️⃣ 메인 엔진 초기화")
    main_engine = MainEngine()
    main_engine.start()
    
    # 2. Gateway 연결
    print("\n2️⃣ Data Gateway 연결")
    kr_gateway = main_engine.add_gateway(KRDataGateway)
    
    if not main_engine.connect_gateway('KRDataGateway'):
        print("  ⚠️ Gateway 연결 실패, 샘플 데이터 사용")
        
        # 샘플 데이터
        dates = pd.date_range('2023-06-01', '2024-01-31', freq='D')
        df = pd.DataFrame({
            'Open': 70000 + np.cumsum(np.random.randn(len(dates)) * 500),
            'High': 70500 + np.cumsum(np.random.randn(len(dates)) * 500),
            'Low': 69500 + np.cumsum(np.random.randn(len(dates)) * 500),
            'Close': 70000 + np.cumsum(np.random.randn(len(dates)) * 500),
            'Volume': np.random.randint(10000000, 50000000, len(dates))
        }, index=dates)
        
        df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(len(dates)) * 200)
        df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(len(dates)) * 200)
    else:
        # 실제 데이터
        df = kr_gateway.get_bars('005930', '2023-06-01', '2024-01-31')
    
    print(f"  데이터 Shape: {df.shape}")
    
    # 3. 팩터 계산
    print("\n3️⃣ 팩터 계산 (Alpha 158)")
    features = Alpha158.generate_basic(df)
    print(f"  팩터 개수: {features.shape[1]}개")
    
    # 4. 신호 생성 (간단한 모멘텀 전략)
    print("\n4️⃣ 신호 생성")
    # 60일 모멘텀
    momentum_60d = features['ROC_60'].iloc[-1]
    # 20일 이동평균 대비 현재가
    ma_20 = features['MA_20'].iloc[-1]
    current_price = features['CLOSE'].iloc[-1]
    
    if momentum_60d > 0.05 and current_price > ma_20:
        signal = "BUY"
        reason = f"모멘텀 {momentum_60d:.2%}, 가격 > MA20"
    elif momentum_60d < -0.05 and current_price < ma_20:
        signal = "SELL"
        reason = f"모멘텀 {momentum_60d:.2%}, 가격 < MA20"
    else:
        signal = "HOLD"
        reason = "조건 미충족"
    
    print(f"  신호: {signal}")
    print(f"  이유: {reason}")
    
    # 5. 이벤트 발행
    print("\n5️⃣ 이벤트 발행")
    main_engine.event_engine.put(Event(
        EventType.SIGNAL_GENERATED,
        {
            "symbol": "005930",
            "signal": signal,
            "momentum": momentum_60d,
            "price": current_price,
            "ma_20": ma_20
        }
    ))
    
    import time
    time.sleep(0.5)
    
    # 6. 정리
    print("\n6️⃣ 시스템 종료")
    main_engine.stop()
    
    print("\n✅ 통합 워크플로우 완료")


def main():
    """메인 함수 - 모든 예제 실행"""
    print("=" * 80)
    print("퀀트 투자 시스템 종합 예제")
    print("GitHub Top 3 프로젝트 (Zipline, VN.py, QuantConnect) 아이디어 적용")
    print("=" * 80)
    
    try:
        # 예제 1: 이벤트 시스템
        example_1_event_system()
        
        # 예제 2: 메인 엔진
        example_2_main_engine()
        
        # 예제 3: Pipeline
        example_3_pipeline()
        
        # 예제 4: Alpha 158
        example_4_alpha158()
        
        # 예제 5: 통합
        example_5_integrated()
        
        print("\n" + "=" * 80)
        print("🎉 모든 예제 완료!")
        print("=" * 80)
        
        print("\n📚 구현된 주요 기능:")
        print("  ✅ 이벤트 엔진 (VN.py 스타일)")
        print("  ✅ Gateway 패턴 (VN.py 스타일)")
        print("  ✅ Pipeline API (Zipline 스타일)")
        print("  ✅ Alpha 158 팩터 (Qlib/VN.py 스타일)")
        print("  ✅ 메인 엔진 통합")
        
        print("\n🚀 다음 단계:")
        print("  1. Zipline 스타일 백테스트 실행")
        print("  2. 실제 전략 구현")
        print("  3. Streamlit UI 개발")
        print("  4. 포트폴리오 최적화")
        
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
