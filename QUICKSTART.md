# 빠른 시작 가이드

## 5분 안에 시작하기

### 1단계: 환경 설정 (1분)

```bash
# 압축 해제
tar -xzf quant_system_v2.tar.gz
cd quant_investment_system_v2

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install pandas numpy python-dateutil
```

### 2단계: 종합 예제 실행 (2분)

```bash
python examples/comprehensive_demo.py
```

이 예제는 다음을 시연합니다:
- ✅ 이벤트 시스템 (VN.py 스타일)
- ✅ Data Gateway (한국/미국 시장)
- ✅ Pipeline API (팩터 계산)
- ✅ Alpha 158 (158개 팩터)
- ✅ 통합 워크플로우

> 참고: 위 명령은 **콘솔 데모**라서 텍스트 로그가 출력되는 것이 정상입니다.

### 2-1단계: 프리미엄 웹 UI 실행 (HTML 대시보드)

```bash
./run_dashboard.sh
# 브라우저에서 http://localhost:8000/dashboard.html
```

### 3단계: 개별 컴포넌트 테스트 (2분)

```bash
# 이벤트 엔진
python engine/event_engine.py

# Gateway 패턴
python data/gateway.py

# Pipeline API
python features/pipeline.py

# Alpha 158
python features/alpha158.py

# 포트폴리오 최적화
python portfolio/optimizer.py

# Purged K-Fold
python validation/purged_kfold.py
```

## 실전 전략 예제

```bash
# 멀티 팩터 전략
python examples/multifactor_strategy.py
```

## 주요 기능 사용법

### 1. 이벤트 시스템

```python
from engine.event_engine import EventEngine, Event, EventType

# 엔진 생성 및 시작
engine = EventEngine()
engine.register(EventType.MARKET_DATA, handler_function)
engine.start()

# 이벤트 발행
engine.put(Event(EventType.MARKET_DATA, {"symbol": "005930", "price": 70000}))
```

### 2. Data Gateway

```python
from engine.main_engine import MainEngine
from data.gateway import KRDataGateway

# 메인 엔진
main_engine = MainEngine()
main_engine.start()

# Gateway 연결
kr_gateway = main_engine.add_gateway(KRDataGateway)
main_engine.connect_gateway('KRDataGateway')

# 데이터 조회
df = kr_gateway.get_bars('005930', '2024-01-01', '2024-01-31')
```

### 3. Pipeline API

```python
from features.pipeline import FactorLibrary

# Pipeline 생성
pipeline = FactorLibrary.create_default_pipeline(
    universe=['AAPL', 'MSFT', 'GOOGL']
)

# 실행
factors = pipeline.run(data, start_date='2024-01-01')
```

### 4. Alpha 158

```python
from features.alpha158 import Alpha158

# 전체 팩터 (158개)
features = Alpha158.generate_all(df)

# 기본 팩터 (26개, 빠름)
basic_features = Alpha158.generate_basic(df)
```

### 5. 포트폴리오 최적화

```python
from portfolio.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()

# 최적화
weights = optimizer.optimize(
    scores=scores,          # 종목 점수
    returns=returns_df,     # 수익률 데이터
    method='inverse_vol'    # 방법
)
```

### 6. Purged K-Fold

```python
from validation.purged_kfold import PurgedKFold

# Purged K-Fold 생성
pkf = PurgedKFold(n_splits=5, purge_days=30, embargo_days=5)

# 교차 검증
for train_idx, test_idx in pkf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # 모델 학습 및 평가
```

## 다음 단계

1. **실제 데이터 연결**: pykrx, yfinance 설치 및 사용
2. **전략 개발**: 자신만의 팩터 및 전략 구현
3. **백테스트**: 과거 데이터로 검증
4. **최적화**: 하이퍼파라미터 튜닝
5. **모니터링**: Streamlit UI 개발

## 문제 해결

### pykrx 설치 오류
```bash
pip install pykrx --break-system-packages
```

### yfinance 설치 오류
```bash
pip install yfinance --upgrade
```

### 일반적인 오류
- Python 3.10 이상 사용 권장
- 가상환경 활성화 확인
- 의존성 재설치: `pip install -r requirements.txt`
- 웹 UI는 `python ui/app.py`가 아니라 `./run_dashboard.sh` 또는 `python -m http.server`로 `ui/dashboard.html`을 열어야 함

## 도움말

- README.md: 전체 개요
- 각 파일 실행: 사용 예제 포함
- examples/: 실전 예제 코드

즐거운 퀀트 트레이딩 되세요! 🚀
