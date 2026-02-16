"""
포트폴리오 최적화
- 변동성 조정 가중 (Inverse Volatility)
- 위험 예산 (Risk Parity)
- 제약조건 적용
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioConstraints:
    """포트폴리오 제약조건"""
    min_position: int = 8        # 최소 종목 수
    max_position: int = 15       # 최대 종목 수
    min_weight: float = 0.03     # 최소 비중 (3%)
    max_weight: float = 0.15     # 최대 비중 (15%)
    max_sector_weight: float = 0.40  # 섹터 최대 비중 (40%)
    max_turnover: float = 0.50   # 최대 회전율 (50%)


class PortfolioOptimizer:
    """
    포트폴리오 최적화
    """
    
    def __init__(self, constraints: Optional[PortfolioConstraints] = None):
        """
        초기화
        
        Args:
            constraints: 제약조건
        """
        self.constraints = constraints or PortfolioConstraints()
        
    def inverse_volatility_weights(
        self,
        returns: pd.DataFrame,
        lookback_days: int = 60
    ) -> pd.Series:
        """
        변동성 역수 가중
        - 변동성이 낮을수록 높은 비중
        
        Args:
            returns: 수익률 데이터프레임 (columns: symbols)
            lookback_days: 변동성 계산 기간
            
        Returns:
            종목별 비중 (합=1)
        """
        # 변동성 계산
        volatilities = returns.tail(lookback_days).std()
        
        # 변동성 역수
        inv_vol = 1 / volatilities
        
        # 정규화 (합=1)
        weights = inv_vol / inv_vol.sum()
        
        return weights
    
    def risk_parity_weights(
        self,
        returns: pd.DataFrame,
        lookback_days: int = 60
    ) -> pd.Series:
        """
        위험 균등 (Risk Parity)
        - 각 자산의 위험 기여도를 동일하게
        
        Args:
            returns: 수익률 데이터프레임
            lookback_days: 공분산 계산 기간
            
        Returns:
            종목별 비중
        """
        # 공분산 행렬
        cov_matrix = returns.tail(lookback_days).cov()
        
        # 초기 가중치 (균등)
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        # 반복 최적화 (간단한 구현)
        for _ in range(10):
            # 포트폴리오 변동성
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            
            # 한계 위험 기여도
            marginal_contrib = cov_matrix @ weights / port_vol
            
            # 위험 기여도
            risk_contrib = weights * marginal_contrib
            
            # 가중치 조정
            weights = weights * (1 / risk_contrib)
            weights = weights / weights.sum()
        
        return pd.Series(weights, index=returns.columns)
    
    def apply_constraints(
        self,
        weights: pd.Series,
        current_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        제약조건 적용
        
        Args:
            weights: 초기 비중
            current_weights: 현재 비중 (회전율 계산용)
            
        Returns:
            조정된 비중
        """
        adjusted = weights.copy()
        
        # 1. 종목 수 제약
        n_positions = (adjusted > 0).sum()
        
        if n_positions > self.constraints.max_position:
            # 상위 max_position 개만 유지
            top_positions = adjusted.nlargest(self.constraints.max_position)
            adjusted = pd.Series(0, index=adjusted.index)
            adjusted[top_positions.index] = top_positions
        
        elif n_positions < self.constraints.min_position:
            # 부족하면 다음 순위 추가
            n_to_add = self.constraints.min_position - n_positions
            zero_positions = adjusted[adjusted == 0].index
            
            if len(zero_positions) >= n_to_add:
                # 0인 것 중 원래 점수가 높은 순으로 추가
                # (여기서는 간단히 균등 분배)
                added_weight = self.constraints.min_weight
                for symbol in zero_positions[:n_to_add]:
                    adjusted[symbol] = added_weight
        
        # 2. 개별 비중 제약
        adjusted = adjusted.clip(
            lower=self.constraints.min_weight,
            upper=self.constraints.max_weight
        )
        
        # 3. 정규화
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum()
        
        # 4. 회전율 제약 (현재 비중이 있는 경우)
        if current_weights is not None:
            turnover = (adjusted - current_weights).abs().sum() / 2
            
            if turnover > self.constraints.max_turnover:
                # 회전율이 높으면 현재 비중 쪽으로 조정
                alpha = self.constraints.max_turnover / turnover
                adjusted = alpha * adjusted + (1 - alpha) * current_weights
                adjusted = adjusted / adjusted.sum()
        
        return adjusted
    
    def optimize(
        self,
        scores: pd.Series,
        returns: pd.DataFrame,
        method: str = 'inverse_vol',
        current_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        포트폴리오 최적화
        
        Args:
            scores: 종목별 점수 (높을수록 좋음)
            returns: 수익률 데이터프레임
            method: 최적화 방법 ('inverse_vol', 'risk_parity', 'equal')
            current_weights: 현재 비중
            
        Returns:
            최적 비중
        """
        # 상위 종목 선택
        top_scores = scores.nlargest(self.constraints.max_position * 2)
        top_symbols = top_scores.index.tolist()
        
        # 해당 종목의 수익률만 사용
        returns_subset = returns[top_symbols].dropna()
        
        if len(returns_subset) < 10:
            print("⚠️ Warning: Insufficient data for optimization")
            # 균등 가중으로 폴백
            weights = pd.Series(1 / len(top_symbols), index=top_symbols)
        else:
            # 방법별 비중 계산
            if method == 'inverse_vol':
                weights = self.inverse_volatility_weights(returns_subset)
            elif method == 'risk_parity':
                weights = self.risk_parity_weights(returns_subset)
            elif method == 'equal':
                weights = pd.Series(1 / len(top_symbols), index=top_symbols)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        # 제약조건 적용
        final_weights = self.apply_constraints(weights, current_weights)
        
        return final_weights


class RegimeBasedAdjustment:
    """
    레짐 기반 비중 조정
    """
    
    @staticmethod
    def detect_regime(
        market_returns: pd.Series,
        vix: Optional[pd.Series] = None,
        lookback: int = 60
    ) -> str:
        """
        시장 레짐 감지
        
        Args:
            market_returns: 시장 수익률
            vix: VIX 지수 (선택)
            lookback: 계산 기간
            
        Returns:
            레짐 ('BULL', 'NORMAL', 'VOLATILE', 'CRISIS')
        """
        recent_returns = market_returns.tail(lookback)
        
        # 수익률
        cumulative_return = (1 + recent_returns).prod() - 1
        
        # 변동성
        volatility = recent_returns.std() * np.sqrt(252)
        
        # 최대 낙폭
        cumulative = (1 + recent_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max).min()
        
        # 레짐 분류
        if cumulative_return > 0.10 and volatility < 0.20:
            return 'BULL'
        elif abs(drawdown) > 0.15 or volatility > 0.35:
            return 'CRISIS'
        elif volatility > 0.25:
            return 'VOLATILE'
        else:
            return 'NORMAL'
    
    @staticmethod
    def adjust_weights(
        weights: pd.Series,
        regime: str,
        constraints: PortfolioConstraints
    ) -> Tuple[pd.Series, float]:
        """
        레짐에 따라 비중 조정
        
        Args:
            weights: 원래 비중
            regime: 시장 레짐
            constraints: 제약조건
            
        Returns:
            (조정된 비중, 현금 비율)
        """
        if regime == 'CRISIS':
            # 위기: 종목 수 최소화, 현금 보유
            target_positions = constraints.min_position
            cash_ratio = 0.20  # 20% 현금
        elif regime == 'VOLATILE':
            # 변동성: 보수적
            target_positions = (constraints.min_position + constraints.max_position) // 2
            cash_ratio = 0.10  # 10% 현금
        elif regime == 'BULL':
            # 강세: 적극적
            target_positions = constraints.max_position
            cash_ratio = 0.0
        else:  # NORMAL
            target_positions = (constraints.min_position + constraints.max_position) // 2
            cash_ratio = 0.0
        
        # 상위 종목만 유지
        top_weights = weights.nlargest(target_positions)
        adjusted = pd.Series(0, index=weights.index)
        adjusted[top_weights.index] = top_weights
        
        # 정규화 (현금 비율 고려)
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum() * (1 - cash_ratio)
        
        return adjusted, cash_ratio


# 사용 예시
if __name__ == "__main__":
    print("=" * 80)
    print("포트폴리오 최적화 예제")
    print("=" * 80)
    
    # 샘플 데이터
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
    
    # 수익률 데이터 생성
    returns_data = {}
    for symbol in symbols:
        returns_data[symbol] = np.random.randn(len(dates)) * 0.02
    
    returns = pd.DataFrame(returns_data, index=dates)
    
    # 종목 점수 (예: 팩터 점수)
    scores = pd.Series({
        'AAPL': 0.85,
        'MSFT': 0.92,
        'GOOGL': 0.78,
        'AMZN': 0.88,
        'TSLA': 0.65,
        'META': 0.82,
        'NVDA': 0.95,
        'JPM': 0.70
    })
    
    print("\n종목 점수:")
    print(scores.sort_values(ascending=False))
    
    # 포트폴리오 최적화
    print("\n" + "=" * 80)
    print("1. 변동성 역수 가중")
    print("=" * 80)
    
    optimizer = PortfolioOptimizer()
    weights_iv = optimizer.optimize(scores, returns, method='inverse_vol')
    
    print("\n비중:")
    print(weights_iv[weights_iv > 0].sort_values(ascending=False))
    print(f"\n종목 수: {(weights_iv > 0).sum()}")
    print(f"총 비중: {weights_iv.sum():.2%}")
    
    # 위험 균등
    print("\n" + "=" * 80)
    print("2. 위험 균등 (Risk Parity)")
    print("=" * 80)
    
    weights_rp = optimizer.optimize(scores, returns, method='risk_parity')
    
    print("\n비중:")
    print(weights_rp[weights_rp > 0].sort_values(ascending=False))
    print(f"\n종목 수: {(weights_rp > 0).sum()}")
    
    # 레짐 기반 조정
    print("\n" + "=" * 80)
    print("3. 레짐 기반 조정")
    print("=" * 80)
    
    # 시장 수익률 (예: S&P 500)
    market_returns = returns.mean(axis=1)
    
    regime = RegimeBasedAdjustment.detect_regime(market_returns)
    print(f"\n현재 레짐: {regime}")
    
    adjusted_weights, cash_ratio = RegimeBasedAdjustment.adjust_weights(
        weights_iv,
        regime,
        optimizer.constraints
    )
    
    print(f"현금 비율: {cash_ratio:.1%}")
    print("\n조정된 비중:")
    print(adjusted_weights[adjusted_weights > 0].sort_values(ascending=False))
    
    # 비교
    print("\n" + "=" * 80)
    print("방법별 비교")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Inverse Vol': weights_iv,
        'Risk Parity': weights_rp,
        f'Adjusted ({regime})': adjusted_weights
    })
    
    print("\n주요 종목 비중 비교:")
    print(comparison[comparison.sum(axis=1) > 0].round(4))
    
    print("\n" + "=" * 80)
    print("✅ 포트폴리오 최적화 완료!")
    print("=" * 80)
