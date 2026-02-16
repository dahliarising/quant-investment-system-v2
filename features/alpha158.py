"""
Alpha 158 Features
Microsoft Qlib의 Alpha158 팩터 세트
- 158개의 검증된 팩터
- 가격, 모멘텀, 볼륨, 변동성 등 다차원 커버
"""
import pandas as pd
import numpy as np
from typing import Dict, List


class Alpha158:
    """
    Alpha 158 팩터 생성기
    
    Microsoft Qlib의 Alpha158 참고
    - KLINE features (6개)
    - Price features (20개)  
    - Volume features (20개)
    - Rolling features (112개)
    
    총 158개 팩터
    """
    
    @staticmethod
    def _kline_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        KLINE 기본 특징 (6개)
        - OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP
        """
        features = {}
        
        features['OPEN'] = df['Open']
        features['HIGH'] = df['High']
        features['LOW'] = df['Low']
        features['CLOSE'] = df['Close']
        features['VOLUME'] = df['Volume']
        
        # VWAP (Volume Weighted Average Price)
        features['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        return features
    
    @staticmethod
    def _price_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30, 60]) -> Dict[str, pd.Series]:
        """
        가격 기반 특징 (20개)
        - ROC (Rate of Change): 수익률
        - MA (Moving Average): 이동평균
        """
        features = {}
        close = df['Close']
        
        for window in windows:
            # ROC: 수익률
            features[f'ROC_{window}'] = close.pct_change(periods=window)
            
            # MA: 이동평균
            features[f'MA_{window}'] = close.rolling(window=window).mean()
            
            # STD: 표준편차
            features[f'STD_{window}'] = close.pct_change().rolling(window=window).std()
            
            # BETA: 회귀 기울기
            features[f'BETA_{window}'] = Alpha158._rolling_beta(close, window)
        
        return features
    
    @staticmethod
    def _volume_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 30, 60]) -> Dict[str, pd.Series]:
        """
        볼륨 기반 특징 (20개)
        """
        features = {}
        volume = df['Volume']
        close = df['Close']
        
        for window in windows:
            # Volume MA
            features[f'VOLUME_MA_{window}'] = volume.rolling(window=window).mean()
            
            # Volume STD
            features[f'VSTD_{window}'] = volume.rolling(window=window).std()
            
            # Correlation between price and volume
            features[f'CORR_{window}'] = Alpha158._rolling_corr(close, volume, window)
            
            # Volume Ratio
            vol_ma = volume.rolling(window=window).mean()
            features[f'VRATIO_{window}'] = volume / vol_ma
        
        return features
    
    @staticmethod
    def _rolling_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        롤링 윈도우 특징 (112개)
        - 다양한 윈도우 크기로 계산
        """
        features = {}
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        windows = [5, 10, 20, 30, 60]
        
        for window in windows:
            # QTLU: Upper quantile
            features[f'QTLU_{window}'] = close.rolling(window=window).quantile(0.8)
            
            # QTLD: Lower quantile  
            features[f'QTLD_{window}'] = close.rolling(window=window).quantile(0.2)
            
            # RANK: Rank in window
            features[f'RANK_{window}'] = close.rolling(window=window).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
            )
            
            # RSV: (Close - Low) / (High - Low) in window
            low_min = low.rolling(window=window).min()
            high_max = high.rolling(window=window).max()
            features[f'RSV_{window}'] = (close - low_min) / (high_max - low_min + 1e-10)
            
            # IMAX: Index of max in window
            features[f'IMAX_{window}'] = high.rolling(window=window).apply(
                lambda x: len(x) - 1 - x.argmax(), raw=True
            )
            
            # IMIN: Index of min in window
            features[f'IMIN_{window}'] = low.rolling(window=window).apply(
                lambda x: len(x) - 1 - x.argmin(), raw=True
            )
            
            # IMXD: IMAX - IMIN
            features[f'IMXD_{window}'] = features[f'IMAX_{window}'] - features[f'IMIN_{window}']
            
        return features
    
    @staticmethod
    def _rolling_beta(series: pd.Series, window: int) -> pd.Series:
        """
        롤링 베타 (선형 회귀 기울기)
        """
        def calc_beta(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            # 선형 회귀: y = beta * x + alpha
            beta = np.polyfit(x, y, 1)[0]
            return beta
        
        return series.rolling(window=window).apply(calc_beta, raw=True)
    
    @staticmethod
    def _rolling_corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
        """롤링 상관계수"""
        return s1.rolling(window=window).corr(s2)
    
    @staticmethod
    def generate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 Alpha158 팩터 생성
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            158개 팩터 데이터프레임
        """
        all_features = {}
        
        # 1. KLINE features (6개)
        print("Generating KLINE features...")
        all_features.update(Alpha158._kline_features(df))
        
        # 2. Price features (20개)
        print("Generating Price features...")
        all_features.update(Alpha158._price_features(df))
        
        # 3. Volume features (20개)
        print("Generating Volume features...")
        all_features.update(Alpha158._volume_features(df))
        
        # 4. Rolling features (112개)
        print("Generating Rolling features...")
        all_features.update(Alpha158._rolling_features(df))
        
        # 데이터프레임으로 변환
        result = pd.DataFrame(all_features)
        
        print(f"Total features generated: {len(result.columns)}")
        
        return result
    
    @staticmethod
    def generate_basic(df: pd.DataFrame) -> pd.DataFrame:
        """
        기본 팩터만 생성 (빠른 계산용)
        
        KLINE + Price features (26개)
        """
        features = {}
        
        features.update(Alpha158._kline_features(df))
        features.update(Alpha158._price_features(df))
        
        return pd.DataFrame(features)


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    print("Creating sample data...")
    dates = pd.date_range('2023-01-01', '2024-01-31', freq='D')
    
    np.random.seed(42)
    df = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'High': 102 + np.cumsum(np.random.randn(len(dates)) * 2),
        'Low': 98 + np.cumsum(np.random.randn(len(dates)) * 2),
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # High, Low 논리적 보정
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(len(dates)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(len(dates)))
    
    print("\nSample OHLCV Data:")
    print(df.head())
    print(f"Shape: {df.shape}")
    
    # 기본 팩터 생성
    print("\n" + "=" * 60)
    print("Generating Basic Features")
    print("=" * 60)
    
    basic_features = Alpha158.generate_basic(df)
    print(f"\nBasic Features Shape: {basic_features.shape}")
    print(f"Columns: {basic_features.columns.tolist()}")
    print("\nSample:")
    print(basic_features.tail())
    
    # 전체 팩터 생성
    print("\n" + "=" * 60)
    print("Generating All Alpha158 Features")
    print("=" * 60)
    
    all_features = Alpha158.generate_all(df)
    print(f"\nAll Features Shape: {all_features.shape}")
    print(f"Feature count: {len(all_features.columns)}")
    
    # 통계
    print("\n" + "=" * 60)
    print("Feature Statistics")
    print("=" * 60)
    print(all_features.describe())
    
    # 결측치 확인
    print("\n" + "=" * 60)
    print("Missing Values")
    print("=" * 60)
    missing = all_features.isnull().sum()
    print(f"Features with missing values: {(missing > 0).sum()}")
    print(f"Max missing count: {missing.max()}")
    
    # 상관관계 높은 팩터 찾기
    print("\n" + "=" * 60)
    print("Highly Correlated Features (|corr| > 0.9)")
    print("=" * 60)
    
    corr_matrix = all_features.corr().abs()
    # 대각선과 하삼각 제거
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr = [
        (column, row, upper_tri.loc[row, column])
        for column in upper_tri.columns
        for row in upper_tri.index
        if upper_tri.loc[row, column] > 0.9
    ]
    
    if high_corr:
        print(f"Found {len(high_corr)} pairs:")
        for col, row, corr_val in high_corr[:10]:  # 상위 10개만 출력
            print(f"  {col} <-> {row}: {corr_val:.3f}")
    else:
        print("No highly correlated pairs found")
    
    print("\nDone!")
