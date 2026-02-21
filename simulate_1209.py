
import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LottoDataLoader
from src.ensemble_predictor import EnsemblePredictor
from src.utils.formatter import LottoFormatter

def simulate_1209():
    print("\n🔍 1209회차 예측 시뮬레이션 (1회 ~ 1208회 학습)")
    print("=" * 60)

    # 1. 데이터 로드
    loader = LottoDataLoader()
    # 전체 매트릭스 가져오기 (1209회차 데이터까지 포함됨)
    full_matrix = loader.get_numbers_matrix()
    
    # 2. 데이터 슬라이싱 (1209회차 제외, 1208회차까지만 사용)
    # full_matrix의 마지막이 1209회차이므로, -1까지 슬라이싱
    train_matrix = full_matrix[:-1]
    
    # 실제 1209회차 정답
    actual_1209 = full_matrix[-1]
    
    print(f"📊 학습 데이터: 1회 ~ {len(train_matrix)}회 (총 {len(train_matrix)}개)")
    print(f"🎯 예측 대상: 1209회차")
    print(f"✅ 1209회 정답 번호: {sorted(actual_1209)}")
    print("-" * 60)
    print("⏳ AI 엔진 학습 및 예측 중... (잠시만 기다려주세요)")

    # 3. 예측기 초기화 및 학습 (1208회차까지의 데이터로만)
    predictor = EnsemblePredictor(train_matrix, use_ml=True, use_validator=True)
    
    # 4. 예측 실행 (다중 세트)
    predicted_sets = predictor.predict_multiple_sets(n_sets=10)
    
    # 5. 결과 분석
    print("\n📈 [분석 결과]")
    
    hit_counts = []
    
    for i, (pred_nums, conf) in enumerate(predicted_sets, 1):
        # 정답 일치 개수 확인
        matches = set(pred_nums) & set(actual_1209)
        hit_count = len(matches)
        hit_counts.append(hit_count)
        
        match_str = ", ".join(map(str, sorted(matches))) if matches else "없음"
        
        print(f"\n[조합 {i}] 신뢰도: {conf:.1f}%")
        print(f"  예측: {pred_nums}")
        print(f"  결과: {hit_count}개 일치 ({match_str})")
        if hit_count >= 3:
            print(f"  🎉 {hit_count}개 적중! (5등 이상)")
            
    # 전체 엔진 예측 요약
    print("\n🔍 [엔진별 추천 번호와 정답 비교]")
    engine_preds = predictor.get_all_predictions()
    for name, nums in engine_preds.items():
        matches = set(nums) & set(actual_1209)
        print(f"  - {name.ljust(20)}: {len(matches)}개 적중 {sorted(matches)}")

if __name__ == "__main__":
    simulate_1209()
