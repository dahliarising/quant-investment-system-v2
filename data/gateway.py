"""
Data Gateway - 데이터 소스 추상화
VN.py의 Gateway 패턴 참고
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

import sys
sys.path.append('/home/claude/quant_investment_system_v2')
from engine.event_engine import EventEngine, Event, EventType


@dataclass
class BarData:
    """봉 데이터"""
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __post_init__(self):
        """유효성 검사"""
        assert self.high >= self.low, "High must be >= Low"
        assert self.high >= self.open, "High must be >= Open"
        assert self.high >= self.close, "High must be >= Close"
        assert self.low <= self.open, "Low must be <= Open"
        assert self.low <= self.close, "Low must be <= Close"


class DataGateway(ABC):
    """
    데이터 게이트웨이 추상 클래스
    - 다양한 데이터 소스 통합
    - 이벤트 기반 데이터 스트리밍
    """
    
    def __init__(self, event_engine: Optional[EventEngine] = None):
        """
        초기화
        
        Args:
            event_engine: 이벤트 엔진 (None이면 이벤트 발행 안함)
        """
        self.event_engine = event_engine
        self.gateway_name = "BaseGateway"
        self.connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """
        데이터 소스에 연결
        
        Returns:
            연결 성공 여부
        """
        pass
    
    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        봉 데이터 조회
        
        Args:
            symbol: 종목 코드
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            interval: 주기 (1d, 1h, 5m 등)
            
        Returns:
            OHLCV DataFrame
        """
        pass
    
    @abstractmethod
    def get_universe(
        self,
        date: str,
        market: str = "KOSPI",
        min_market_cap: Optional[float] = None
    ) -> List[str]:
        """
        투자 유니버스 조회
        
        Args:
            date: 날짜
            market: 시장 (KOSPI, KOSDAQ 등)
            min_market_cap: 최소 시가총액
            
        Returns:
            종목 코드 리스트
        """
        pass
    
    def on_bar(self, bar: BarData):
        """
        봉 데이터 수신 시 호출
        - 이벤트 발행
        """
        if self.event_engine:
            event = Event(
                EventType.MARKET_DATA,
                {"bar": bar, "gateway": self.gateway_name}
            )
            self.event_engine.put(event)
    
    def on_bars(self, bars: pd.DataFrame):
        """
        여러 봉 데이터 수신 시 호출
        """
        if self.event_engine:
            event = Event(
                EventType.HISTORICAL_DATA,
                {"bars": bars, "gateway": self.gateway_name}
            )
            self.event_engine.put(event)
    
    def close(self):
        """연결 종료"""
        self.connected = False


class KRDataGateway(DataGateway):
    """
    한국 시장 데이터 게이트웨이
    - pykrx 기반
    """
    
    def __init__(self, event_engine: Optional[EventEngine] = None):
        super().__init__(event_engine)
        self.gateway_name = "KRDataGateway"
        
    def connect(self) -> bool:
        """연결 (pykrx는 별도 연결 불필요)"""
        try:
            import pykrx
            self.connected = True
            return True
        except ImportError:
            print("pykrx not installed. Run: pip install pykrx")
            return False
    
    def get_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        봉 데이터 조회
        
        Note: pykrx는 일봉만 지원
        """
        if not self.connected:
            self.connect()
        
        try:
            from pykrx import stock
            
            # OHLCV 데이터 조회
            df = stock.get_market_ohlcv_by_date(
                start_date.replace('-', ''),
                end_date.replace('-', ''),
                symbol
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 컬럼명 표준화
            df = df.rename(columns={
                '시가': 'Open',
                '고가': 'High',
                '저가': 'Low',
                '종가': 'Close',
                '거래량': 'Volume'
            })
            
            # 인덱스를 datetime으로 변환
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            # 이벤트 발행
            self.on_bars(df)
            
            return df
            
        except Exception as e:
            print(f"Error getting bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_universe(
        self,
        date: str,
        market: str = "KOSPI",
        min_market_cap: Optional[float] = None
    ) -> List[str]:
        """투자 유니버스 조회"""
        if not self.connected:
            self.connect()
        
        try:
            from pykrx import stock
            
            # 시장별 종목 리스트
            if market == "KOSPI":
                tickers = stock.get_market_ticker_list(
                    date.replace('-', ''),
                    market="KOSPI"
                )
            elif market == "KOSDAQ":
                tickers = stock.get_market_ticker_list(
                    date.replace('-', ''),
                    market="KOSDAQ"
                )
            else:
                tickers = stock.get_market_ticker_list(
                    date.replace('-', '')
                )
            
            # 시가총액 필터링
            if min_market_cap:
                market_caps = stock.get_market_cap_by_ticker(
                    date.replace('-', ''),
                    market=market
                )
                filtered_tickers = [
                    ticker for ticker in tickers
                    if ticker in market_caps.index 
                    and market_caps.loc[ticker, '시가총액'] >= min_market_cap
                ]
                return filtered_tickers
            
            return tickers
            
        except Exception as e:
            print(f"Error getting universe: {e}")
            return []


class USDataGateway(DataGateway):
    """
    미국 시장 데이터 게이트웨이
    - yfinance 기반
    """
    
    def __init__(self, event_engine: Optional[EventEngine] = None):
        super().__init__(event_engine)
        self.gateway_name = "USDataGateway"
        
    def connect(self) -> bool:
        """연결"""
        try:
            import yfinance
            self.connected = True
            return True
        except ImportError:
            print("yfinance not installed. Run: pip install yfinance")
            return False
    
    def get_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """봉 데이터 조회"""
        if not self.connected:
            self.connect()
        
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 컬럼명 표준화
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # 필요한 컬럼만 선택
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # 이벤트 발행
            self.on_bars(df)
            
            return df
            
        except Exception as e:
            print(f"Error getting bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_universe(
        self,
        date: str,
        market: str = "US",
        min_market_cap: Optional[float] = None
    ) -> List[str]:
        """
        투자 유니버스 조회
        
        Note: yfinance는 유니버스 조회 기능이 없어서
        미리 정의된 리스트 사용
        """
        # S&P 500 주요 종목 (예시)
        sp500_major = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'V', 'JNJ',
            'WMT', 'JPM', 'MA', 'PG', 'UNH',
            'HD', 'DIS', 'BAC', 'ADBE', 'CRM'
        ]
        return sp500_major


# 사용 예시
if __name__ == "__main__":
    from engine.event_engine import EventEngine, EventType
    
    # 이벤트 엔진 생성
    event_engine = EventEngine()
    
    # 핸들러 등록
    def on_historical_data(event):
        bars = event.data['bars']
        gateway = event.data['gateway']
        print(f"\n[{gateway}] Historical data received:")
        print(bars.head())
        print(f"Shape: {bars.shape}")
    
    event_engine.register(EventType.HISTORICAL_DATA, on_historical_data)
    event_engine.start()
    
    # 한국 시장 게이트웨이 테스트
    print("=" * 60)
    print("Testing KR Data Gateway")
    print("=" * 60)
    kr_gateway = KRDataGateway(event_engine)
    
    if kr_gateway.connect():
        # 삼성전자 데이터 조회
        df = kr_gateway.get_bars('005930', '2024-01-01', '2024-01-31')
        
        # 유니버스 조회
        universe = kr_gateway.get_universe('2024-01-31', 'KOSPI', min_market_cap=1e12)
        print(f"\nKOSPI Universe (시총 1조 이상): {len(universe)}개 종목")
        print(f"Sample: {universe[:5]}")
    
    # 미국 시장 게이트웨이 테스트
    print("\n" + "=" * 60)
    print("Testing US Data Gateway")
    print("=" * 60)
    us_gateway = USDataGateway(event_engine)
    
    if us_gateway.connect():
        # Apple 데이터 조회
        df = us_gateway.get_bars('AAPL', '2024-01-01', '2024-01-31')
        
        # 유니버스 조회
        universe = us_gateway.get_universe('2024-01-31')
        print(f"\nUS Universe: {len(universe)}개 종목")
        print(f"Sample: {universe[:5]}")
    
    # 정리
    import time
    time.sleep(1)
    event_engine.stop()
    print("\nDone!")
