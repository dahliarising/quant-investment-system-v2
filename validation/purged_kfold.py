"""
Purged K-Fold Cross Validation
- 시계열 데이터의 미래 정보 누수 방지
- Embargo 기간 포함
- Advances in Financial Machine Learning (Marcos Lopez de Prado) 참고
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import timedelta


class PurgedKFold:
    """
    Purged K-Fold Cross Validation
    
    시계열 데이터에서 학습/검증 세트 간 정보 누수를 방지하기 위해:
    1. 검증 세트 직전의 학습 데이터를 제거 (Purge)
    2. 검증 세트 직후에 추가 갭 설정 (Embargo)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 30,
        embargo_days: int = 5
    ):
        """
        초기화
        
        Args:
            n_splits: fold 개수
            purge_days: 제거할 기간 (일)
            embargo_days: 추가 갭 기간 (일)
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        학습/검증 인덱스 생성
        
        Args:
            X: 특징 데이터프레임 (index는 datetime)
            y: 타겟 (사용 안함, sklearn 호환성)
            
        Returns:
            (train_indices, test_indices) 리스트
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        
        dates = X.index.unique().sort_values()
        n_samples = len(dates)
        
        # 각 fold의 크기
        fold_size = n_samples // self.n_splits
        
        splits = []
        
        for fold in range(self.n_splits):
            # 검증 세트 범위
            test_start_idx = fold * fold_size
            test_end_idx = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            
            test_start_date = dates[test_start_idx]
            test_end_date = dates[test_end_idx - 1]
            
            # Purge 기간 계산
            purge_start_date = test_start_date - timedelta(days=self.purge_days)
            
            # Embargo 기간 계산
            embargo_end_date = test_end_date + timedelta(days=self.embargo_days)
            
            # 학습 세트: purge 이전 + embargo 이후
            train_mask = (
                (dates < purge_start_date) |  # purge 이전
                (dates > embargo_end_date)     # embargo 이후
            )
            
            # 검증 세트
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)
            
            # 인덱스 변환
            train_dates = dates[train_mask]
            test_dates = dates[test_mask]
            
            train_indices = X.index.isin(train_dates)
            test_indices = X.index.isin(test_dates)
            
            # numpy array로 변환
            train_idx = np.where(train_indices)[0]
            test_idx = np.where(test_indices)[0]
            
            splits.append((train_idx, test_idx))
            
            # 정보 출력
            print(f"\nFold {fold + 1}/{self.n_splits}:")
            print(f"  Train: {len(train_idx):,} samples")
            print(f"    Before purge: {train_dates[train_dates < purge_start_date].min()} ~ "
                  f"{train_dates[train_dates < purge_start_date].max()}")
            if len(train_dates[train_dates > embargo_end_date]) > 0:
                print(f"    After embargo: {train_dates[train_dates > embargo_end_date].min()} ~ "
                      f"{train_dates[train_dates > embargo_end_date].max()}")
            print(f"  Test:  {len(test_idx):,} samples ({test_start_date.date()} ~ {test_end_date.date()})")
            print(f"  Purge period: {self.purge_days} days")
            print(f"  Embargo period: {self.embargo_days} days")
        
        return splits
    
    def get_n_splits(self) -> int:
        """Fold 개수 반환"""
        return self.n_splits


class TimeSeriesSplit:
    """
    시계열 분할 (확장 윈도우)
    - 학습 세트가 점점 커짐
    - 검증 세트는 항상 미래
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: Optional[int] = None,
        test_size: Optional[int] = None
    ):
        """
        초기화
        
        Args:
            n_splits: fold 개수
            min_train_size: 최소 학습 크기
            test_size: 검증 크기 (None이면 자동)
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """학습/검증 인덱스 생성"""
        n_samples = len(X)
        
        # 검증 크기 결정
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        # 최소 학습 크기
        if self.min_train_size is None:
            min_train_size = test_size
        else:
            min_train_size = self.min_train_size
        
        splits = []
        
        for i in range(self.n_splits):
            # 검증 세트
            test_start = min_train_size + (i * test_size)
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            # 학습 세트 (처음부터 검증 직전까지)
            train_idx = np.arange(0, test_start)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            splits.append((train_idx, test_idx))
            
            print(f"\nFold {i + 1}/{self.n_splits}:")
            print(f"  Train: {len(train_idx):,} samples (index 0 ~ {test_start - 1})")
            print(f"  Test:  {len(test_idx):,} samples (index {test_start} ~ {test_end - 1})")
        
        return splits


class ValidationMetrics:
    """검증 메트릭스"""
    
    @staticmethod
    def rank_ic(predictions: pd.Series, actuals: pd.Series) -> float:
        """
        Rank IC (Information Coefficient)
        - 예측값과 실제값의 Spearman 상관계수
        
        Args:
            predictions: 예측값
            actuals: 실제값
            
        Returns:
            Rank IC
        """
        return predictions.corr(actuals, method='spearman')
    
    @staticmethod
    def hit_rate(predictions: pd.Series, actuals: pd.Series) -> float:
        """
        Hit Rate (방향 정확도)
        
        Args:
            predictions: 예측값
            actuals: 실제값
            
        Returns:
            Hit rate (0~1)
        """
        pred_direction = (predictions > 0).astype(int)
        actual_direction = (actuals > 0).astype(int)
        
        return (pred_direction == actual_direction).mean()
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe Ratio
        
        Args:
            returns: 수익률 시리즈
            risk_free_rate: 무위험 수익률 (연율)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def max_drawdown(cumulative_returns: pd.Series) -> float:
        """
        Maximum Drawdown
        
        Args:
            cumulative_returns: 누적 수익률
            
        Returns:
            Maximum drawdown (음수)
        """
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    print("=" * 80)
    print("Purged K-Fold Cross Validation 예제")
    print("=" * 80)
    
    # 날짜 범위
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    
    # 특징 데이터
    np.random.seed(42)
    X = pd.DataFrame({
        'feature_1': np.random.randn(len(dates)),
        'feature_2': np.random.randn(len(dates)),
        'feature_3': np.random.randn(len(dates))
    }, index=dates)
    
    # 타겟 (30일 후 수익률)
    y = pd.Series(np.random.randn(len(dates)) * 0.1, index=dates)
    
    print(f"\n데이터 크기: {len(X):,} samples")
    print(f"기간: {X.index.min().date()} ~ {X.index.max().date()}")
    
    # Purged K-Fold
    print("\n" + "=" * 80)
    print("Purged K-Fold (n_splits=5, purge=30일, embargo=5일)")
    print("=" * 80)
    
    pkf = PurgedKFold(n_splits=5, purge_days=30, embargo_days=5)
    splits = pkf.split(X)
    
    # 각 fold의 성능 평가 (시뮬레이션)
    print("\n" + "=" * 80)
    print("Fold별 성능 평가 (시뮬레이션)")
    print("=" * 80)
    
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 간단한 선형 모델 시뮬레이션
        # 실제로는 LightGBM 등 사용
        predictions = y_test + np.random.randn(len(y_test)) * 0.05
        
        # 메트릭스 계산
        rank_ic = ValidationMetrics.rank_ic(predictions, y_test)
        hit_rate = ValidationMetrics.hit_rate(predictions, y_test)
        
        print(f"\nFold {i + 1}:")
        print(f"  Rank IC: {rank_ic:.4f}")
        print(f"  Hit Rate: {hit_rate:.4f}")
    
    # Time Series Split
    print("\n" + "=" * 80)
    print("Time Series Split (확장 윈도우)")
    print("=" * 80)
    
    tss = TimeSeriesSplit(n_splits=5, min_train_size=365)
    ts_splits = tss.split(X)
    
    # 검증 세트 크기 비교
    print("\n" + "=" * 80)
    print("검증 방법 비교")
    print("=" * 80)
    
    print("\nPurged K-Fold:")
    print(f"  - 학습 세트 크기: 변동 (purge/embargo 제외)")
    print(f"  - 검증 세트 크기: 균등")
    print(f"  - 미래 정보 누수: 방지됨")
    print(f"  - 용도: 금융 시계열 데이터")
    
    print("\nTime Series Split:")
    print(f"  - 학습 세트 크기: 점진적 증가")
    print(f"  - 검증 세트 크기: 고정")
    print(f"  - 미래 정보 누수: 방지됨")
    print(f"  - 용도: 순차적 시계열 예측")
    
    print("\n" + "=" * 80)
    print("✅ Purged K-Fold 구현 완료!")
    print("=" * 80)
    
    print("\n💡 실전 사용 예시:")
    print("""
from validation.purged_kfold import PurgedKFold, ValidationMetrics
from lightgbm import LGBMRegressor

# Purged K-Fold 생성
pkf = PurgedKFold(n_splits=5, purge_days=30, embargo_days=5)

# 교차 검증
ic_scores = []

for train_idx, test_idx in pkf.split(X):
    # 학습/검증 분할
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 모델 학습
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    
    # 예측
    predictions = model.predict(X_test)
    
    # 평가
    ic = ValidationMetrics.rank_ic(pd.Series(predictions), y_test)
    ic_scores.append(ic)

# 평균 IC
mean_ic = np.mean(ic_scores)
print(f"Mean Rank IC: {mean_ic:.4f}")
    """)
