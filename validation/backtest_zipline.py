"""
Zipline 스타일 백테스트 엔진
- initialize/handle_data 패턴
- 직관적인 전략 작성
"""
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    shares: int
    avg_price: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """현재 평가액"""
        return self.shares * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """취득원가"""
        return self.shares * self.avg_price
    
    @property
    def unrealized_pnl(self) -> float:
        """미실현 손익"""
        return self.market_value - self.cost_basis


@dataclass 
class Portfolio:
    """포트폴리오 상태"""
    cash: float = 100_000_000  # 초기 현금 1억
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def positions_value(self) -> float:
        """포지션 총 평가액"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def portfolio_value(self) -> float:
        """총 포트폴리오 가치"""
        return self.cash + self.positions_value
    
    @property
    def returns(self) -> float:
        """수익률"""
        initial_value = 100_000_000
        return (self.portfolio_value - initial_value) / initial_value


class Context:
    """
    전략 컨텍스트
    - 전략 실행 중 상태 유지
    """
    
    def __init__(self):
        self.portfolio = Portfolio()
        self.data_frequency = 'daily'
        self.current_date: Optional[datetime] = None
        
        # 사용자 정의 변수 저장
        self._user_vars: Dict[str, Any] = {}
    
    def __getattr__(self, name: str):
        """사용자 변수 접근"""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return self._user_vars.get(name)
    
    def __setattr__(self, name: str, value: Any):
        """사용자 변수 설정"""
        if name in ['portfolio', 'data_frequency', 'current_date', '_user_vars']:
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_user_vars'):
                super().__setattr__('_user_vars', {})
            self._user_vars[name] = value


class DataPortal:
    """
    데이터 포털
    - 백테스트용 데이터 제공
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        초기화
        
        Args:
            data: OHLCV 데이터 (MultiIndex: date, symbol)
        """
        self.data = data
        
    def get_current_data(self, date: datetime, symbols: List[str]) -> pd.DataFrame:
        """
        현재 시점 데이터 조회
        
        Args:
            date: 날짜
            symbols: 종목 리스트
            
        Returns:
            해당 날짜의 데이터
        """
        try:
            current_data = self.data.xs(date, level='date')
            if symbols:
                current_data = current_data[current_data.index.isin(symbols)]
            return current_data
        except KeyError:
            return pd.DataFrame()
    
    def get_history(
        self,
        symbol: str,
        fields: List[str],
        bar_count: int,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        과거 데이터 조회
        
        Args:
            symbol: 종목 코드
            fields: 필드 리스트 (예: ['Close', 'Volume'])
            bar_count: 봉 개수
            end_date: 종료일
            
        Returns:
            과거 데이터
        """
        try:
            symbol_data = self.data.xs(symbol, level='symbol')
            symbol_data = symbol_data[symbol_data.index <= end_date]
            return symbol_data[fields].tail(bar_count)
        except KeyError:
            return pd.DataFrame()


class BacktestEngine:
    """
    백테스트 엔진
    - Zipline 스타일 API
    """
    
    def __init__(
        self,
        initialize: Callable,
        handle_data: Callable,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        capital_base: float = 100_000_000
    ):
        """
        초기화
        
        Args:
            initialize: 초기화 함수 (context)
            handle_data: 데이터 처리 함수 (context, data)
            data: OHLCV 데이터 (MultiIndex: date, symbol)
            start_date: 시작일
            end_date: 종료일
            capital_base: 초기 자본
        """
        self.initialize = initialize
        self.handle_data = handle_data
        self.data = data
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.capital_base = capital_base
        
        # 컨텍스트 및 데이터 포털
        self.context = Context()
        self.context.portfolio.cash = capital_base
        self.data_portal = DataPortal(data)
        
        # 실행 기록
        self.performance: List[Dict] = []
        
    def order_target(self, symbol: str, target_shares: int, price: float):
        """
        목표 주식 수량으로 주문
        
        Args:
            symbol: 종목 코드
            target_shares: 목표 주식 수
            price: 현재 가격
        """
        portfolio = self.context.portfolio
        
        # 현재 보유량
        current_shares = 0
        if symbol in portfolio.positions:
            current_shares = portfolio.positions[symbol].shares
        
        # 매수/매도 수량
        delta_shares = target_shares - current_shares
        
        if delta_shares == 0:
            return
        
        # 거래 금액
        trade_value = abs(delta_shares) * price
        commission = trade_value * 0.0015  # 0.15% 수수료
        
        if delta_shares > 0:  # 매수
            cost = trade_value + commission
            if cost > portfolio.cash:
                # 현금 부족 - 가능한 만큼만 매수
                available_cash = portfolio.cash
                affordable_shares = int(available_cash / (price * 1.0015))
                delta_shares = affordable_shares
                cost = delta_shares * price * 1.0015
            
            if delta_shares <= 0:
                return
            
            portfolio.cash -= cost
            
            # 포지션 업데이트
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                total_shares = pos.shares + delta_shares
                total_cost = pos.cost_basis + delta_shares * price
                pos.shares = total_shares
                pos.avg_price = total_cost / total_shares
            else:
                portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    shares=delta_shares,
                    avg_price=price
                )
                
        else:  # 매도
            if symbol not in portfolio.positions:
                return
            
            pos = portfolio.positions[symbol]
            sell_shares = min(abs(delta_shares), pos.shares)
            
            proceeds = sell_shares * price - commission
            portfolio.cash += proceeds
            
            pos.shares -= sell_shares
            
            # 포지션 청산
            if pos.shares == 0:
                del portfolio.positions[symbol]
    
    def order_target_percent(self, symbol: str, target_percent: float, price: float):
        """
        포트폴리오 비율로 주문
        
        Args:
            symbol: 종목 코드
            target_percent: 목표 비율 (0.0 ~ 1.0)
            price: 현재 가격
        """
        portfolio = self.context.portfolio
        target_value = portfolio.portfolio_value * target_percent
        target_shares = int(target_value / price)
        self.order_target(symbol, target_shares, price)
    
    def update_positions(self, current_prices: pd.Series):
        """포지션 현재가 업데이트"""
        for symbol, pos in self.context.portfolio.positions.items():
            if symbol in current_prices.index:
                pos.current_price = current_prices[symbol]
    
    def record_performance(self, date: datetime):
        """성과 기록"""
        portfolio = self.context.portfolio
        
        record = {
            'date': date,
            'portfolio_value': portfolio.portfolio_value,
            'cash': portfolio.cash,
            'positions_value': portfolio.positions_value,
            'returns': portfolio.returns,
            'num_positions': len(portfolio.positions)
        }
        
        self.performance.append(record)
    
    def run(self) -> pd.DataFrame:
        """
        백테스트 실행
        
        Returns:
            성과 데이터프레임
        """
        print(f"Running backtest from {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial capital: ${self.capital_base:,.0f}")
        
        # 초기화 함수 실행
        self.initialize(self.context)
        
        # 날짜 범위
        trading_dates = self.data.index.get_level_values('date').unique()
        trading_dates = trading_dates[
            (trading_dates >= self.start_date) & (trading_dates <= self.end_date)
        ]
        
        # 각 거래일마다 실행
        for i, date in enumerate(trading_dates):
            self.context.current_date = date
            
            # 현재 데이터 조회
            current_data = self.data_portal.get_current_data(date, symbols=None)
            
            if current_data.empty:
                continue
            
            # 포지션 업데이트
            self.update_positions(current_data['Close'])
            
            # handle_data 함수 실행
            self.handle_data(self.context, self.data_portal)
            
            # 성과 기록
            self.record_performance(date)
            
            # 진행 상황 출력
            if (i + 1) % 50 == 0 or i == len(trading_dates) - 1:
                pct_complete = (i + 1) / len(trading_dates) * 100
                pv = self.context.portfolio.portfolio_value
                ret = self.context.portfolio.returns
                print(f"  [{pct_complete:5.1f}%] {date.date()}: "
                      f"Portfolio=${pv:,.0f}, Return={ret:+.2%}")
        
        # 결과 데이터프레임
        result = pd.DataFrame(self.performance)
        result.set_index('date', inplace=True)
        
        # 최종 통계
        print("\n" + "=" * 60)
        print("Backtest Results")
        print("=" * 60)
        print(f"Final Portfolio Value: ${result['portfolio_value'].iloc[-1]:,.0f}")
        print(f"Total Return: {result['returns'].iloc[-1]:+.2%}")
        print(f"Max Drawdown: {self._calculate_max_drawdown(result):.2%}")
        print(f"Sharpe Ratio: {self._calculate_sharpe(result):.2f}")
        
        return result
    
    def _calculate_max_drawdown(self, result: pd.DataFrame) -> float:
        """최대 낙폭 계산"""
        portfolio_value = result['portfolio_value']
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe(self, result: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        returns = result['portfolio_value'].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252  # 일간 무위험 수익률
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    print("Creating sample data...")
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=['date', 'symbol']
    )
    
    np.random.seed(42)
    data = pd.DataFrame({
        'Open': 100 + np.random.randn(len(index)) * 10,
        'High': 105 + np.random.randn(len(index)) * 10,
        'Low': 95 + np.random.randn(len(index)) * 10,
        'Close': 100 + np.random.randn(len(index)) * 10,
        'Volume': np.random.randint(1000000, 10000000, len(index))
    }, index=index)
    
    # Close가 범위 내에 있도록 조정
    data['Close'] = data[['High', 'Low']].mean(axis=1)
    
    # 전략 정의
    def initialize(context):
        """전략 초기화"""
        print("\nInitializing strategy...")
        context.stocks = ['AAPL', 'MSFT', 'GOOGL']
        context.rebalance_frequency = 20  # 20일마다 리밸런싱
        context.last_rebalance = 0
    
    def handle_data(context, data):
        """매 거래일마다 실행"""
        # 리밸런싱 체크
        context.last_rebalance += 1
        
        if context.last_rebalance < context.rebalance_frequency:
            return
        
        context.last_rebalance = 0
        
        # 현재 데이터 조회
        current_date = context.current_date
        
        # 간단한 모멘텀 전략: 최근 60일 수익률
        rankings = []
        
        for symbol in context.stocks:
            hist = data.get_history(symbol, ['Close'], bar_count=60, end_date=current_date)
            
            if len(hist) < 60:
                continue
            
            # 60일 수익률
            momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            rankings.append((symbol, momentum))
        
        # 모멘텀 순으로 정렬
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 2종목에 균등 투자
        top_stocks = [symbol for symbol, _ in rankings[:2]]
        
        # 현재 가격
        current_data = data.get_current_data(current_date, symbols=top_stocks)
        
        # 리밸런싱
        target_weight = 0.4  # 각 종목 40% (나머지 20%는 현금)
        
        for symbol in context.stocks:
            if symbol in top_stocks and symbol in current_data.index:
                price = current_data.loc[symbol, 'Close']
                # Backtest 엔진에서 order_target_percent 호출
                context.portfolio  # 포트폴리오 접근용
            else:
                # 보유하지 않을 종목은 청산
                pass
    
    # 백테스트 실행
    print("\n" + "=" * 60)
    print("Starting Backtest")
    print("=" * 60)
    
    engine = BacktestEngine(
        initialize=initialize,
        handle_data=handle_data,
        data=data,
        start_date='2023-01-01',
        end_date='2023-12-31',
        capital_base=100_000_000
    )
    
    result = engine.run()
    
    print("\nPerformance DataFrame:")
    print(result.tail(10))
    
    print("\nDone!")
