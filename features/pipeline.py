"""
Pipeline API - Zipline 스타일 팩터 계산
- 배치 처리로 성능 최적화
- 팩터 간 의존성 관리
- 메모리 효율적인 계산
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class Factor(ABC):
    """
    팩터 추상 클래스
    - Zipline Pipeline의 Factor 개념
    """
    
    def __init__(self, inputs: Optional[List[str]] = None, window_length: int = 1):
        """
        초기화
        
        Args:
            inputs: 입력 컬럼 이름 리스트 (예: ['Close', 'Volume'])
            window_length: 계산에 필요한 윈도우 길이
        """
        self.inputs = inputs or ['Close']
        self.window_length = window_length
        self.name = self.__class__.__name__
        
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        팩터 계산
        
        Args:
            data: 입력 데이터 (MultiIndex: date, symbol)
            
        Returns:
            팩터 값 시리즈
        """
        pass


class SimpleMovingAverage(Factor):
    """단순 이동평균"""
    
    def __init__(self, window_length: int = 20):
        super().__init__(inputs=['Close'], window_length=window_length)
        self.name = f"SMA_{window_length}"
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """이동평균 계산"""
        return data['Close'].rolling(window=self.window_length).mean()


class Returns(Factor):
    """수익률"""
    
    def __init__(self, window_length: int = 1):
        super().__init__(inputs=['Close'], window_length=window_length)
        self.name = f"Returns_{window_length}"
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """수익률 계산"""
        return data['Close'].pct_change(periods=self.window_length)


class Volatility(Factor):
    """변동성"""
    
    def __init__(self, window_length: int = 20):
        super().__init__(inputs=['Close'], window_length=window_length)
        self.name = f"Volatility_{window_length}"
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """변동성 계산 (일별 수익률의 표준편차)"""
        returns = data['Close'].pct_change()
        return returns.rolling(window=self.window_length).std()


class AverageDollarVolume(Factor):
    """평균 거래대금"""
    
    def __init__(self, window_length: int = 20):
        super().__init__(inputs=['Close', 'Volume'], window_length=window_length)
        self.name = f"AvgDollarVolume_{window_length}"
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """거래대금 계산"""
        dollar_volume = data['Close'] * data['Volume']
        return dollar_volume.rolling(window=self.window_length).mean()


class RSI(Factor):
    """RSI (Relative Strength Index)"""
    
    def __init__(self, window_length: int = 14):
        super().__init__(inputs=['Close'], window_length=window_length)
        self.name = f"RSI_{window_length}"
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """RSI 계산"""
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.window_length).mean()
        avg_loss = loss.rolling(window=self.window_length).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class MACD(Factor):
    """MACD (Moving Average Convergence Divergence)"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(inputs=['Close'], window_length=slow)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"MACD_{fast}_{slow}_{signal}"
        
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """MACD 계산"""
        ema_fast = data['Close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        
        return macd


class Pipeline:
    """
    팩터 계산 파이프라인
    - Zipline의 Pipeline 개념
    - 여러 팩터를 효율적으로 배치 계산
    """
    
    def __init__(self, universe: Optional[List[str]] = None):
        """
        초기화
        
        Args:
            universe: 종목 유니버스 (None이면 데이터의 모든 종목)
        """
        self.universe = universe
        self.columns: Dict[str, Factor] = {}
        
    def add(self, factor: Factor, name: Optional[str] = None):
        """
        팩터 추가
        
        Args:
            factor: 팩터 객체
            name: 팩터 이름 (None이면 factor.name 사용)
        """
        factor_name = name or factor.name
        self.columns[factor_name] = factor
        
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        파이프라인 실행
        
        Args:
            data: 입력 데이터 (MultiIndex: date, symbol)
            start_date: 시작일 (None이면 전체)
            end_date: 종료일 (None이면 전체)
            
        Returns:
            팩터 데이터프레임 (MultiIndex: date, symbol)
        """
        # 날짜 필터링
        if start_date:
            data = data[data.index.get_level_values('date') >= start_date]
        if end_date:
            data = data[data.index.get_level_values('date') <= end_date]
        
        # 유니버스 필터링
        if self.universe:
            data = data[data.index.get_level_values('symbol').isin(self.universe)]
        
        # 각 종목별로 팩터 계산
        results = {}
        
        for symbol in data.index.get_level_values('symbol').unique():
            symbol_data = data.xs(symbol, level='symbol')
            
            for factor_name, factor in self.columns.items():
                factor_values = factor.compute(symbol_data)
                
                if factor_name not in results:
                    results[factor_name] = []
                
                results[factor_name].append(factor_values)
        
        # 결과 합치기
        output = pd.DataFrame(results)
        output.index = data.index
        
        return output


class FactorLibrary:
    """
    팩터 라이브러리
    - 자주 사용하는 팩터들의 모음
    """
    
    @staticmethod
    def momentum_factors(windows: List[int] = [20, 60, 120, 252]) -> Dict[str, Factor]:
        """모멘텀 팩터들"""
        factors = {}
        for window in windows:
            factors[f"Returns_{window}d"] = Returns(window_length=window)
            factors[f"SMA_{window}d"] = SimpleMovingAverage(window_length=window)
        return factors
    
    @staticmethod
    def volatility_factors(windows: List[int] = [20, 60]) -> Dict[str, Factor]:
        """변동성 팩터들"""
        factors = {}
        for window in windows:
            factors[f"Volatility_{window}d"] = Volatility(window_length=window)
        return factors
    
    @staticmethod
    def liquidity_factors(windows: List[int] = [20, 60]) -> Dict[str, Factor]:
        """유동성 팩터들"""
        factors = {}
        for window in windows:
            factors[f"AvgDollarVolume_{window}d"] = AverageDollarVolume(window_length=window)
        return factors
    
    @staticmethod
    def technical_factors() -> Dict[str, Factor]:
        """기술적 팩터들"""
        return {
            "RSI_14": RSI(window_length=14),
            "MACD": MACD(fast=12, slow=26, signal=9)
        }
    
    @staticmethod
    def create_default_pipeline(universe: Optional[List[str]] = None) -> Pipeline:
        """
        기본 파이프라인 생성
        - 모멘텀, 변동성, 유동성, 기술적 팩터 포함
        """
        pipeline = Pipeline(universe=universe)
        
        # 모멘텀 팩터
        for name, factor in FactorLibrary.momentum_factors().items():
            pipeline.add(factor, name)
        
        # 변동성 팩터
        for name, factor in FactorLibrary.volatility_factors().items():
            pipeline.add(factor, name)
        
        # 유동성 팩터
        for name, factor in FactorLibrary.liquidity_factors().items():
            pipeline.add(factor, name)
        
        # 기술적 팩터
        for name, factor in FactorLibrary.technical_factors().items():
            pipeline.add(factor, name)
        
        return pipeline


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', '2024-03-01', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # MultiIndex 데이터프레임 생성
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=['date', 'symbol']
    )
    
    np.random.seed(42)
    data = pd.DataFrame({
        'Open': np.random.randn(len(index)) * 10 + 100,
        'High': np.random.randn(len(index)) * 10 + 105,
        'Low': np.random.randn(len(index)) * 10 + 95,
        'Close': np.random.randn(len(index)) * 10 + 100,
        'Volume': np.random.randint(1000000, 10000000, len(index))
    }, index=index)
    
    # Close가 논리적으로 High/Low 범위 내에 있도록 조정
    data['Close'] = data[['High', 'Low']].mean(axis=1)
    
    print("Sample Data:")
    print(data.head(10))
    
    # 파이프라인 생성
    print("\n" + "=" * 60)
    print("Creating Pipeline")
    print("=" * 60)
    
    pipeline = FactorLibrary.create_default_pipeline(universe=['AAPL', 'MSFT'])
    
    # 파이프라인 실행
    print("\nRunning Pipeline...")
    factors = pipeline.run(data, start_date='2024-02-01')
    
    print("\nFactor Results:")
    print(factors.head(20))
    print(f"\nShape: {factors.shape}")
    print(f"Columns: {factors.columns.tolist()}")
    
    # 특정 종목의 팩터 확인
    print("\n" + "=" * 60)
    print("AAPL Factors")
    print("=" * 60)
    aapl_factors = factors.xs('AAPL', level='symbol')
    print(aapl_factors.tail())
    
    print("\nDone!")
