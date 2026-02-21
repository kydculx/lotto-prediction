
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LottoDataLoader
from src.ensemble_predictor import EnsemblePredictor

def verify_enhanced_predictor():
    print("=" * 60)
    print("🚀 고도화된 앙상블 예측기 v3.1 검증 시작")
    print("=" * 60)
    
    loader = LottoDataLoader()
    loader.load()
    matrix = loader.get_numbers_matrix()
    
    print(f"\n[1] 데이터 로드 완료: {len(matrix)}회차")
    
    print("\n[2] 예측기 초기화 (신규 엔진 로드 및 동적 가중치 계산 포함)...")
    # ML 엔진은 테스트 속도를 위해 제외할 수 있으나, 여기선 전체 로드
    predictor = EnsemblePredictor(train_matrix, use_ml=True, use_validator=True) 
    
    report = predictor.get_detailed_report(n_sets=3)
    
    print("\n[3] 신규 엔진 분석 결과:")
    for eng in ['poisson', 'fourier']:
        pred = report['engine_predictions'].get(eng, [])
        print(f"  - {eng.capitalize()} 엔진 예측: {pred}")
        
    print("\n[4] 동적 가중치 분석 (Meta-Learning):")
    print(f"  {'엔진명':<20} | {'부스트':<10} | {'최종 가중치':<10}")
    print("-" * 50)
    
    weights = report['final_weights']
    boosts = report['dynamic_boosts']
    
    for name in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
        boost = boosts.get(name, 1.0)
        weight = weights[name]
        print(f"  {name:<20} | {boost:9.2f} | {weight:13.4f}")

    print("\n[5] 최종 추천 조합 (Top 3):")
    for i, (nums, conf) in enumerate(report['predicted_sets'], 1):
        print(f"  SET {i}: {nums} (신뢰도: {conf:.1f}%)")

if __name__ == "__main__":
    verify_enhanced_predictor()
