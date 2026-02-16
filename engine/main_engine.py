"""
메인 엔진 - VN.py 스타일
모든 컴포넌트를 통합하는 중앙 엔진
"""
from typing import Dict, List, Optional, Type
import sys
sys.path.append('/home/claude/quant_investment_system_v2')

from engine.event_engine import EventEngine, EventType, Event
from data.gateway import DataGateway, KRDataGateway, USDataGateway


class MainEngine:
    """
    메인 엔진
    - 모든 컴포넌트의 중앙 관리자
    - VN.py의 MainEngine 참고
    """
    
    def __init__(self, event_engine: Optional[EventEngine] = None):
        """
        초기화
        
        Args:
            event_engine: 이벤트 엔진 (None이면 새로 생성)
        """
        self.event_engine = event_engine or EventEngine()
        
        # 게이트웨이 저장소
        self.gateways: Dict[str, DataGateway] = {}
        
        # 앱 저장소 (향후 확장용)
        self.apps: Dict[str, any] = {}
        
        # 엔진 상태
        self.is_running = False
        
    def add_gateway(self, gateway_class: Type[DataGateway], name: Optional[str] = None):
        """
        게이트웨이 추가
        
        Args:
            gateway_class: 게이트웨이 클래스
            name: 게이트웨이 이름 (None이면 클래스명 사용)
        """
        gateway_name = name or gateway_class.__name__
        
        # 게이트웨이 인스턴스 생성
        gateway = gateway_class(event_engine=self.event_engine)
        
        # 저장
        self.gateways[gateway_name] = gateway
        
        print(f"Gateway added: {gateway_name}")
        
        return gateway
    
    def get_gateway(self, name: str) -> Optional[DataGateway]:
        """게이트웨이 조회"""
        return self.gateways.get(name)
    
    def connect_gateway(self, name: str) -> bool:
        """게이트웨이 연결"""
        gateway = self.get_gateway(name)
        
        if not gateway:
            print(f"Gateway not found: {name}")
            return False
        
        success = gateway.connect()
        
        if success:
            print(f"Gateway connected: {name}")
        else:
            print(f"Gateway connection failed: {name}")
        
        return success
    
    def add_app(self, app_class: Type, name: Optional[str] = None):
        """
        앱 추가 (향후 확장용)
        
        Args:
            app_class: 앱 클래스
            name: 앱 이름
        """
        app_name = name or app_class.__name__
        
        # 앱 인스턴스 생성 (향후 구현)
        print(f"App added: {app_name} (not implemented yet)")
    
    def start(self):
        """엔진 시작"""
        if self.is_running:
            print("MainEngine is already running")
            return
        
        print("Starting MainEngine...")
        
        # 이벤트 엔진 시작
        self.event_engine.start()
        
        self.is_running = True
        print("MainEngine started successfully")
    
    def stop(self):
        """엔진 정지"""
        if not self.is_running:
            print("MainEngine is not running")
            return
        
        print("Stopping MainEngine...")
        
        # 모든 게이트웨이 연결 종료
        for name, gateway in self.gateways.items():
            gateway.close()
            print(f"Gateway closed: {name}")
        
        # 이벤트 엔진 정지
        self.event_engine.stop()
        
        self.is_running = False
        print("MainEngine stopped")
    
    def write_log(self, msg: str, level: str = "INFO"):
        """로그 작성"""
        event = Event(
            EventType.LOG,
            {"message": msg, "level": level}
        )
        self.event_engine.put(event)


# 사용 예시
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Main Engine Example")
    print("=" * 60)
    
    # 메인 엔진 생성
    main_engine = MainEngine()
    
    # 로그 핸들러 등록
    def on_log(event: Event):
        data = event.data
        print(f"[{data['level']}] {data['message']}")
    
    main_engine.event_engine.register(EventType.LOG, on_log)
    
    # 데이터 핸들러 등록
    def on_historical_data(event: Event):
        bars = event.data['bars']
        gateway = event.data['gateway']
        print(f"\n[DATA] Received from {gateway}:")
        print(f"  Shape: {bars.shape}")
        print(f"  Date range: {bars.index.min()} ~ {bars.index.max()}")
    
    main_engine.event_engine.register(EventType.HISTORICAL_DATA, on_historical_data)
    
    # 엔진 시작
    main_engine.start()
    
    # 게이트웨이 추가
    kr_gateway = main_engine.add_gateway(KRDataGateway, "Korea")
    us_gateway = main_engine.add_gateway(USDataGateway, "US")
    
    # 게이트웨이 연결
    main_engine.connect_gateway("Korea")
    main_engine.connect_gateway("US")
    
    # 로그 작성
    main_engine.write_log("System initialized")
    
    # 데이터 조회 테스트
    print("\n" + "=" * 60)
    print("Testing Data Retrieval")
    print("=" * 60)
    
    # 한국 시장 데이터
    if kr_gateway.connected:
        main_engine.write_log("Fetching Korean market data...")
        df_kr = kr_gateway.get_bars('005930', '2024-01-01', '2024-01-31')
        
        if not df_kr.empty:
            print(f"\n삼성전자 (005930) 데이터:")
            print(df_kr.tail())
    
    # 미국 시장 데이터
    if us_gateway.connected:
        main_engine.write_log("Fetching US market data...")
        df_us = us_gateway.get_bars('AAPL', '2024-01-01', '2024-01-31')
        
        if not df_us.empty:
            print(f"\nApple (AAPL) 데이터:")
            print(df_us.tail())
    
    # 잠시 대기
    print("\n" + "=" * 60)
    print("Running for 2 seconds...")
    print("=" * 60)
    time.sleep(2)
    
    # 엔진 정지
    main_engine.stop()
    
    print("\nDone!")
