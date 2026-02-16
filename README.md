# 개선된 퀀트 투자 시스템 v2.0

GitHub Top 3 퀀트 프로젝트의 모범 사례를 적용한 실전 투자 시스템

## 🎯 주요 특징

### 1. 이벤트 기반 아키텍처 (VN.py 참고)
- **EventEngine**: pub/sub 패턴으로 모듈 간 느슨한 결합
- **비동기 처리**: 멀티스레드 이벤트 처리
- **확장성**: 새로운 모듈 쉽게 추가 가능

### 2. Gateway 패턴 (VN.py 참고)
- **데이터 소스 추상화**: pykrx, yfinance 통합
- **이벤트 발행**: 데이터 수신 시 자동 이벤트 발행
- **확장 가능**: 새로운 데이터 소스 쉽게 추가

### 3. Pipeline API (Zipline 참고)
- **배치 처리**: 여러 팩터를 효율적으로 계산
- **메모리 효율**: 중복 계산 제거
- **병렬 처리 지원**: 대규모 데이터 처리 최적화

### 4. Alpha 158 팩터 (Microsoft Qlib 참고)
- **158개 검증된 팩터**: 가격, 모멘텀, 볼륨, 변동성
- **산업 표준**: 실전에서 검증된 팩터 세트
- **즉시 사용 가능**: 별도 연구 없이 바로 활용

### 5. Zipline 스타일 백테스트
- **직관적 API**: initialize/handle_data 패턴
- **쉬운 전략 작성**: 몇 줄의 코드로 전략 구현
- **완전한 백테스트**: 수수료, 슬리피지 포함

## 📁 프로젝트 구조

```
quant_investment_system_v2/
├── engine/                 # 핵심 엔진
│   ├── event_engine.py     # 이벤트 엔진 (VN.py 스타일)
│   └── main_engine.py      # 메인 엔진 (통합 관리)
├── data/                   # 데이터 레이어
│   ├── gateway.py          # Gateway 패턴 (KR/US)
│   └── providers/          # 데이터 프로바이더
├── features/               # 팩터 엔진
│   ├── pipeline.py         # Pipeline API (Zipline 스타일)
│   ├── alpha158.py         # Alpha 158 팩터 (Qlib)
│   └── factors/            # 개별 팩터들
├── validation/             # 검증 & 백테스트
│   └── backtest_zipline.py # Zipline 스타일 백테스트
├── portfolio/              # 포트폴리오 관리
├── services/               # 비즈니스 로직
├── ui/                     # Premium Web UI (dashboard.html)
└── examples/               # 사용 예제
    └── comprehensive_demo.py
```

## 🚀 빠른 시작

### 설치

```bash
# 1. 저장소 클론
git clone <repository_url>
cd quant_investment_system_v2

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

### 사용 예제

#### 1. 이벤트 시스템

```python
from engine.event_engine import EventEngine, Event, EventType

# 이벤트 엔진 생성
engine = EventEngine()

# 핸들러 등록
def on_market_data(event):
    print(f"Data: {event.data}")

engine.register(EventType.MARKET_DATA, on_market_data)

# 엔진 시작
engine.start()

# 이벤트 발행
engine.put(Event(EventType.MARKET_DATA, {"symbol": "005930", "price": 70000}))
```

#### 2. Data Gateway

```python
from engine.main_engine import MainEngine
from data.gateway import KRDataGateway

# 메인 엔진 생성
main_engine = MainEngine()
main_engine.start()

# Gateway 추가 및 연결
kr_gateway = main_engine.add_gateway(KRDataGateway)
main_engine.connect_gateway('KRDataGateway')

# 데이터 조회
df = kr_gateway.get_bars('005930', '2024-01-01', '2024-01-31')
```

#### 3. Pipeline API

```python
from features.pipeline import FactorLibrary

# 기본 Pipeline 생성
pipeline = FactorLibrary.create_default_pipeline(
    universe=['AAPL', 'MSFT', 'GOOGL']
)

# 실행
factors = pipeline.run(data, start_date='2024-01-01')
```

#### 4. Alpha 158 팩터

```python
from features.alpha158 import Alpha158

# OHLCV 데이터 준비
df = ...  # 가격 데이터

# 전체 팩터 생성 (158개)
features = Alpha158.generate_all(df)

# 또는 기본 팩터만 (26개, 빠름)
basic_features = Alpha158.generate_basic(df)
```

#### 5. Zipline 스타일 백테스트

```python
from validation.backtest_zipline import BacktestEngine

def initialize(context):
    context.stocks = ['AAPL', 'MSFT']
    context.rebalance_freq = 20

def handle_data(context, data):
    # 전략 로직
    pass

# 백테스트 실행
engine = BacktestEngine(
    initialize=initialize,
    handle_data=handle_data,
    data=data,
    start_date='2023-01-01',
    end_date='2023-12-31'
)

result = engine.run()
```

### 종합 예제 실행

```bash
python examples/comprehensive_demo.py
```

`examples/comprehensive_demo.py`는 콘솔 기반 통합 데모이며 텍스트 로그 출력이 정상입니다.

### 웹 대시보드 실행 (권장)

```bash
./run_dashboard.sh
# 브라우저에서 http://localhost:8000/dashboard.html
```

## 📊 구현된 기능

### ✅ 완료
- [x] 이벤트 엔진 (Event-driven architecture)
- [x] 메인 엔진 (Central management)
- [x] Data Gateway (KR, US markets)
- [x] Pipeline API (Batch factor computation)
- [x] Alpha 158 Features (158 proven factors)
- [x] Zipline-style Backtesting

### 🚧 진행 중
- [ ] Purged K-Fold Validation
- [ ] Portfolio Optimization
- [ ] Risk Management
- [x] Premium Web UI Dashboard (HTML + Plotly)
- [ ] SHAP Explainability

### 📅 계획
- [ ] Live Trading Support
- [ ] Multi-strategy Support
- [ ] Performance Analytics
- [ ] Cloud Deployment

## 🎓 참고 프로젝트

이 시스템은 다음 오픈소스 프로젝트들의 아이디어를 참고했습니다:

1. **Zipline (Quantopian)** ⭐ 19.2k stars
   - Pipeline API
   - Event-driven backtesting
   - initialize/handle_data pattern

2. **VN.py (VeighNa)** ⭐ 32.9k stars
   - Event Engine architecture
   - Gateway pattern
   - Main Engine design

3. **Microsoft Qlib**
   - Alpha 158 factor library
   - AI-driven quantitative strategies

## 📝 라이선스

MIT License

## 🤝 기여

이슈 및 풀 리퀘스트 환영합니다!

## 📧 문의

프로젝트 관련 문의사항은 이슈로 등록해주세요.
