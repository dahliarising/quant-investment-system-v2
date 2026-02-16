"""
이벤트 엔진 - VN.py 스타일
모듈 간 느슨한 결합을 위한 pub/sub 패턴
"""
from collections import defaultdict
from queue import Queue, Empty
from threading import Thread
from typing import Callable, Any, Dict, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import time


class EventType(Enum):
    """이벤트 타입"""
    # 데이터 이벤트
    MARKET_DATA = "market_data"
    HISTORICAL_DATA = "historical_data"
    
    # 팩터 이벤트
    FACTOR_CALCULATED = "factor_calculated"
    FACTOR_UPDATE = "factor_update"
    
    # 신호 이벤트
    SIGNAL_GENERATED = "signal_generated"
    RANKING_UPDATE = "ranking_update"
    
    # 포트폴리오 이벤트
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    POSITION_UPDATE = "position_update"
    
    # 레짐 이벤트
    REGIME_CHANGE = "regime_change"
    
    # 시스템 이벤트
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    TIMER = "timer"
    LOG = "log"


@dataclass
class Event:
    """이벤트 클래스"""
    type: EventType
    data: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventEngine:
    """
    이벤트 엔진
    - 비동기 이벤트 처리
    - pub/sub 패턴
    - 멀티스레드 지원
    """
    
    def __init__(self):
        """초기화"""
        self._queue = Queue()
        self._active = False
        self._thread = Thread(target=self._run)
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._general_handlers: List[Callable] = []
        
    def start(self):
        """엔진 시작"""
        self._active = True
        self._thread.start()
        self.put(Event(EventType.SYSTEM_START, {"message": "EventEngine started"}))
        
    def stop(self):
        """엔진 정지"""
        self._active = False
        self.put(Event(EventType.SYSTEM_STOP, {"message": "EventEngine stopping"}))
        self._thread.join()
        
    def put(self, event: Event):
        """이벤트 발행"""
        self._queue.put(event)
        
    def register(self, event_type: EventType, handler: Callable):
        """
        특정 이벤트 타입에 핸들러 등록
        
        Args:
            event_type: 이벤트 타입
            handler: 핸들러 함수 (event를 인자로 받음)
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            
    def unregister(self, event_type: EventType, handler: Callable):
        """핸들러 등록 해제"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            
    def register_general(self, handler: Callable):
        """모든 이벤트를 받는 일반 핸들러 등록"""
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)
            
    def _run(self):
        """이벤트 처리 루프 (별도 스레드에서 실행)"""
        while self._active:
            try:
                event = self._queue.get(timeout=1)
                self._process(event)
            except Empty:
                continue
                
    def _process(self, event: Event):
        """이벤트 처리"""
        # 특정 타입 핸들러 실행
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in handler {handler.__name__}: {e}")
                    
        # 일반 핸들러 실행
        for handler in self._general_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in general handler {handler.__name__}: {e}")


class TimerEngine:
    """
    타이머 엔진
    - 주기적 이벤트 발생
    """
    
    def __init__(self, event_engine: EventEngine):
        """초기화"""
        self.event_engine = event_engine
        self._active = False
        self._thread = Thread(target=self._run)
        self._interval = 1.0  # 1초
        
    def start(self):
        """타이머 시작"""
        self._active = True
        self._thread.start()
        
    def stop(self):
        """타이머 정지"""
        self._active = False
        self._thread.join()
        
    def _run(self):
        """타이머 루프"""
        while self._active:
            event = Event(EventType.TIMER, {"time": datetime.now()})
            self.event_engine.put(event)
            time.sleep(self._interval)


# 사용 예시
if __name__ == "__main__":
    # 이벤트 엔진 생성
    engine = EventEngine()
    
    # 핸들러 정의
    def on_market_data(event: Event):
        print(f"Market data received: {event.data}")
    
    def on_signal(event: Event):
        print(f"Signal generated: {event.data}")
    
    def on_all_events(event: Event):
        print(f"[ALL] {event.type.value}: {event.timestamp}")
    
    # 핸들러 등록
    engine.register(EventType.MARKET_DATA, on_market_data)
    engine.register(EventType.SIGNAL_GENERATED, on_signal)
    engine.register_general(on_all_events)
    
    # 엔진 시작
    engine.start()
    
    # 이벤트 발행
    engine.put(Event(EventType.MARKET_DATA, {"symbol": "005930", "price": 70000}))
    engine.put(Event(EventType.SIGNAL_GENERATED, {"symbol": "005930", "action": "BUY"}))
    
    # 잠시 대기
    time.sleep(2)
    
    # 엔진 정지
    engine.stop()
    print("Done!")
