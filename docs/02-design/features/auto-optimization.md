# PDCA Design: Auto Optimization & Training System

> **Feature**: Auto Weight Optimization
> **Related Plan**: LOTTO-PREDICT-2026
> **Status**: Implemented
> **File**: `auto_optimize.py`, `train_1000.py`

## 1. Overview (개요)
예측 시스템의 정확도를 유지하기 위해 가장 중요한 것은 **각 엔진의 가중치(Weight)**를 결정하는 것입니다. 특정 시기에는 '통계적 방법'이 잘 맞고, 다른 시기에는 '패턴 분석'이 잘 맞을 수 있습니다. 본 모듈은 최근 회차(예: 10~100회)의 시뮬레이션을 통해 가장 성적이 좋은 엔진에 높은 가중치를 부여하는 **자동 최적화 로직**을 설계합니다.

## 2. Architecture (아키텍처)

### Flow Chart
```mermaid
graph TD
    A[Start Optimization] --> B{Load History Data}
    B --> C[Run Backtest (Recent N rounds)]
    C --> D[Calculate Engine Accuracy]
    D --> E[Adjust Weights (Boost/Penalty)]
    E --> F[Validation Test]
    F --> G[Save Weights to JSON]
```

### Key Components

#### 1. `auto_optimize.py`
- **역할**: 가중치 최적화의 메인 컨트롤러
- **주요 파라미터**:
    - `--apply`: 최적화된 결과를 실제 설정 파일(`trained_weights.json`)에 적용
    - `--test-rounds`: 테스트할 최근 회차 수 (기본 100)
- **알고리즘**:
    1. 현재 가중치 로드
    2. 최근 N회차에 대해 각 엔진별 단독 예측 수행
    3. 실제 당첨 번호와 비교하여 엔진별 점수(Score) 산출
    4. 점수 기반으로 가중치 재분배 (Softmax 또는 비율 기반 정규화)

#### 2. `train_1000.py`
- **역할**: 대규모 데이터셋(1000회차 이상)을 이용한 ML/DL 모델 학습
- **출력**: 학습된 모델 파일 (`.h5`, `.pkl`) 및 초기 가중치 설정

#### 3. `apply_trained_weights.py`
- **역할**: `auto_optimize.py`에서 생성된 JSON 가중치 파일을 `EnsemblePredictor` 클래스 코드 내의 `DEFAULT_WEIGHTS` 딕셔너리에 주입하거나, 별도 설정 파일로 분리하여 로드하도록 지원.

## 3. Data Structure (데이터 구조)

### Weights JSON Format
```json
{
  "statistical": 0.2319,
  "sequence_correlation": 0.1652,
  "timeseries": 0.1330,
  "advanced_pattern": 0.1225,
  "pattern": 0.0972,
  "lstm": 0.0963,
  "ml": 0.0909,
  "graph": 0.0166,
  ...
}
```

## 4. Optimization Logic (최적화 로직 상세)
- **Moving Average**: 단순 최근 성적이 아닌, 최근 5주, 10주, 20주 성적에 가중치를 달리하여 반영 (최신 경향 중시)
- **Exploration vs Exploitation**: 성적이 나쁜 엔진에도 최소 가중치(Epsilon)를 부여하여, 향후 성능 향상 시 다시 채택될 기회를 제공

## 5. Usage (사용법)
```bash
# 1. 최근 100회차 기준으로 최적화 실행 및 적용
python3 auto_optimize.py --apply --test-rounds 100

# 2. 결과 확인
cat trained_weights_1000.json
```
