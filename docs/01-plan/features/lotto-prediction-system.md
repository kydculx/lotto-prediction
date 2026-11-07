# PDCA Plan: Lotto Prediction System Renewal

> **Project Code**: LOTTO-PREDICT-2026
> **Version**: v3.0
> **Date**: 2026-02-18
> **Author**: Gemini CLI (bkit)

## 1. Project Overview (프로젝트 개요)
본 프로젝트는 **9개 이상의 분석 엔진**과 **ML(Machine Learning)**, **딥러닝(LSTM)** 모델을 결합한 하이브리드 로또 번호 예측 시스템입니다. 과거 1000회차 이상의 데이터를 학습하여 당첨 번호를 예측하며, **동적 가중치 최적화(Dynamic Weighting)** 시스템을 통해 매주 최적의 예측 성능을 유지하는 것을 목표로 합니다.

### Core Goals (핵심 목표)
- **정확도 향상**: 다양한 통계적 기법과 AI 모델의 앙상블을 통한 예측 적중률 극대화
- **자동화**: 매주 새로운 데이터 크롤링부터 학습, 가중치 최적화, 예측 생성까지의 프로세스 자동화
- **확장성**: 새로운 예측 엔진(플러그인)을 손쉽게 추가할 수 있는 유연한 아키텍처 구축

## 2. Key Features (주요 기능)

### A. Ensemble Prediction Engine (앙상블 예측 엔진)
- **다중 엔진 결합**: Statistical, Sequence Correlation, LSTM, ML(XGBoost/LGBM 등), Pattern 등 다양한 엔진의 결과 종합
- **Weighted Voting**: 각 엔진의 신뢰도에 따른 가중치 투표 시스템
- **Combination Validator**: 생성된 조합의 유효성 검증 (AC값, 합계 범위 등)

### B. Auto Optimization System (자동 최적화 시스템)
- **Dynamic Weighting**: 최근 회차의 성적을 바탕으로 각 엔진의 가중치 자동 조절 (`auto_optimize.py`)
- **Performance Tracking**: 엔진별 적중률 모니터링 및 저성과 엔진 가중치 감소
- **Self-Correction**: 예측 실패 시 다음 회차 가중치에 반영하는 피드백 루프

### C. Data Pipeline (데이터 파이프라인)
- **Crawler**: 동행복권 사이트에서 최신 당첨 결과 자동 수집 (`crawler.py`)
- **Data Loader**: 수집된 데이터를 분석 가능한 Matrix 형태로 변환 및 캐싱 (`data_loader.py`)

## 3. Technology Stack (기술 스택)
- **Language**: Python 3.9+
- **Web Framework**: Flask (API & Web Interface)
- **Data Analysis**: NumPy, Pandas
- **AI/ML**: TensorFlow/Keras (LSTM), Scikit-learn, XGBoost
- **Interface**: CLI (`main.py`) & Web (`app.py`)

## 4. Development Roadmap (개발 로드맵)
- [x] Phase 1: 기본 통계 엔진 및 데이터 파이프라인 구축
- [x] Phase 2: LSTM 및 ML 모델 통합
- [x] Phase 3: 앙상블 시스템 및 동적 가중치 구현 (Current)
- [ ] Phase 4: 웹 인터페이스 고도화 및 시각화 (Planned)
- [ ] Phase 5: 백엔드 API 서버 구축 (bkend.ai 연동 예정)

## 5. Constraints & Risks (제약 사항 및 리스크)
- **성능 이슈**: 매 요청 시 무거운 모델 로딩으로 인한 지연 (Refactoring 필요)
- **데이터 의존성**: 외부 사이트 구조 변경 시 크롤러 오작동 가능성
