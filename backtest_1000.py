#!/usr/bin/env python3
"""
백테스팅: 1~1000회차 학습, 1001~1209회차 테스트
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LottoDataLoader
from src.ensemble_predictor import EnsemblePredictor
from src.utils.formatter import LottoFormatter
import numpy as np
import json
import os


def main():
    print("\n" + "=" * 60)
    print("📊 백테스팅: 1~1000회차 학습 → 1001~1209회차 테스트")
    print("=" * 60)
    
    # 데이터 로드
    print("\n⏳ 데이터 로딩...")
    loader = LottoDataLoader()
    full_matrix = loader.get_numbers_matrix()
    df = loader.df
    
    # 1~1000회차만 학습 데이터로 사용
    train_matrix = full_matrix[:1000]
    
    print(f"✅ 학습 데이터: 1~1000회차 (총 {len(train_matrix)}개)")
    print(f"🎯 테스트 데이터: 1001~{len(full_matrix)}회차 (총 {len(full_matrix) - 1000}개)")
    print("\n⏳ AI 모델 학습 중...")
    
    # 가중치 파일 로드
    weights_path = Path(__file__).parent / "trained_weights_1000.json"
    trained_weights = None
    
    if weights_path.exists():
        try:
            with open(weights_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                trained_weights = data.get('weights')
                score = data.get('best_score', 0)
            print(f"📂 학습된 가중치 로드 완료 (점수: {score:.4f})")
        except Exception as e:
            print(f"⚠️ 가중치 로드 실패: {e}")
            
    print("✅ 학습 완료! 테스트 시작...\n")
    print("-" * 60)
    
    # 1001회차부터 테스트
    hit_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    total_hits = 0
    test_count = 0
    
    # Walk-Forward Validation을 위한 현재 데이터 매트릭스
    current_matrix = np.array(train_matrix)
    
    for test_idx in range(1000, len(full_matrix)):
        # 실제 정답
        actual = set(full_matrix[test_idx])
        round_num = int(df.iloc[test_idx]['round'])
        
        # 📌 핵심: 매 회차마다 업데이트된 데이터로 예측기 새로 생성 (미래 정보 반영)
        # 1001회차 예측엔 1~1000회 데이터 사용
        # 1002회차 예측엔 1~1001회 데이터 사용...
        predictor = EnsemblePredictor(
            current_matrix, 
            weights=trained_weights, 
            use_ml=True,        # 정확도를 위해 ML 사용
            use_validator=True  # 정확도를 위해 검증기 사용
        )
        
        # 5개 세트 예측
        predicted_sets = predictor.predict_multiple_sets(5)
        
        # 5개 중 가장 잘 맞은 것 기준
        best_hit = 0
        best_set = None
        
        for pred, _ in predicted_sets:
            hits = len(set(pred) & actual)
            if hits > best_hit:
                best_hit = hits
                best_set = pred
        
        if best_set is None:
            best_set = predicted_sets[0][0]
        
        hit_counts[best_hit] += 1
        total_hits += best_hit
        test_count += 1
        
        # 실시간 로그
        clean_pred = [int(n) for n in best_set]
        clean_actual = [int(n) for n in actual]
        print(f"[{round_num}회차] 최고 적중: {best_hit}개 | 예측: {sorted(clean_pred)} | 정답: {sorted(clean_actual)}")
        
        # 📌 다음 예측을 위해 정답을 데이터에 추가 (재학습 효과)
        # full_matrix[test_idx]는 1차원 배열이므로 2차원으로 변환 후 추가
        new_row = full_matrix[test_idx].reshape(1, 6)
        current_matrix = np.vstack([current_matrix, new_row])
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📈 최종 결과")
    print("=" * 60)
    print(f"\n총 테스트 회차: {test_count}회")
    print(f"평균 적중 개수: {total_hits / test_count:.2f}개\n")
    
    print("적중 분포:")
    for hits in range(6, -1, -1):
        count = hit_counts[hits]
        if count > 0:
            pct = count / test_count * 100
            bar = "█" * int(pct / 2)
            prize = ""
            if hits == 6:
                prize = " (1등!)"
            elif hits == 5:
                prize = " (3등)"
            elif hits == 4:
                prize = " (4등)"
            elif hits == 3:
                prize = " (5등)"
            print(f"  {hits}개 적중: {count:3d}회 ({pct:5.1f}%) {bar}{prize}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
