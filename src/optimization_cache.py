import numpy as np
from typing import List, Dict, Tuple
from src.ensemble_predictor import EnsemblePredictor
import time
import sys

class OptimizationCache:
    """
    유전 알고리즘 속도 향상을 위한 예측 결과 캐싱 클래스 (Vectorized)
    NumPy 행렬 연산을 통해 평가 속도를 극대화함.
    """
    
    def __init__(self):
        # 3D Array: (n_rounds, n_engines, 45) - float64 for precision
        self.cached_params = None 
        self.actual_matrix = None # (n_rounds, 45) - binary
        self.engine_indices = {}
        self.engine_names = []
    
    def precalculate(self, matrix: np.ndarray, test_rounds: int):
        """
        테스트 구간의 모든 엔진 예측값을 미리 계산하여 3D 행렬로 변환
        """
        n_draws = len(matrix)
        
        # 임시 예측기 생성으로 엔진 목록 확인
        temp_predictor = EnsemblePredictor(matrix[:100], use_ml=True, use_validator=True, use_dynamic_weight=True)
        self.engine_names = sorted(list(temp_predictor.engines.keys()))
        self.engine_indices = {name: i for i, name in enumerate(self.engine_names)}
        n_engines = len(self.engine_names)
        
        # 캐시 행렬 초기화 (float64로 정밀도 유지)
        self.cached_scores = np.zeros((test_rounds, n_engines, 45), dtype=np.float64)
        self.actual_matrix = np.zeros((test_rounds, 45), dtype=np.int8)
        
        print(f"\n⚡️ 최적화 캐시 생성 중... (Method: Vectorized Matrix, 총 {test_rounds}회차)")
        start_time = time.time()
        
        # 반복 구간
        for i in range(test_rounds):
            test_idx = n_draws - test_rounds + i
            train_matrix = matrix[:test_idx]
            
            # 실제 당첨 번호 저장
            actual = matrix[test_idx]
            for num in actual:
                self.actual_matrix[i, num - 1] = 1
            
            # 예측기 생성
            predictor = EnsemblePredictor(train_matrix, use_ml=True, use_validator=True, use_dynamic_weight=True)
            
            # 엔진별 점수 계산
            scores = predictor.calculate_all_scores()
            
            # 행렬에 채우기
            for name, engine_score in scores.items():
                if name in self.engine_indices:
                    idx = self.engine_indices[name]
                    for num, score in engine_score.items():
                        self.cached_scores[i, idx, num - 1] = score
            
            # 진행상황 표시
            if (i + 1) % 10 == 0:
                percent = (i + 1) / test_rounds * 100
                sys.stdout.write(f"\r   캐싱 진행률: {percent:5.1f}% ({i+1}/{test_rounds})")
                sys.stdout.flush()
                
        elapsed = time.time() - start_time
        print(f"\n✅ 캐싱 완료! 소요시간: {elapsed:.2f}초\n")
        
        return {
            'scores': self.cached_scores,
            'actuals': self.actual_matrix,
            'engine_names': self.engine_names
        }

    @staticmethod
    def evaluate_weights(cached_data: Dict, weights: Dict[str, float]) -> Tuple[float, Dict[int, int]]:
        """
        Vectorized 평가 함수 for float64 precision
        """
        cached_scores = cached_data['scores'] # (R, E, 45)
        actuals = cached_data['actuals']      # (R, 45)
        engine_names = cached_data['engine_names']
        
        # 가중치 벡터 생성 (float64)
        W = np.zeros(len(engine_names), dtype=np.float64)
        total_weight = 0.0
        
        for name, weight in weights.items():
            if weight > 0 and name in engine_names:
                total_weight += weight
        
        if total_weight == 0:
            return 0.0, {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
            
        # 정규화하여 벡터에 할당
        for i, name in enumerate(engine_names):
            if name in weights:
                W[i] = weights[name] / total_weight
        
        # 1. 앙상블 점수 계산 (행렬 곱)
        # cached_scores shape: (R, E, 45) n_rounds, n_engines, 45_numbers
        # W shape: (E,)
        # We need to contract axis 1 (E) with W.
        # np.dot contracts the last axis (45) which causes mismatch.
        # Use tensordot to specify axes: (R, E, 45) x (E) -> sum over E -> (R, 45)
        ensemble_scores = np.tensordot(cached_scores, W, axes=([1], [0]))
        
        # 2. 상위 6개 선정 (argsort 사용)
        # argsort는 오름차순이므로 뒤에서 6개를 가져옴
        top6_indices = np.argsort(ensemble_scores, axis=1)[:, -6:]
        
        # 3. 적중 개수 계산 (Vectorized)
        # 예측된 인덱스로 binary 마스크 생성
        R = ensemble_scores.shape[0]
        hits = np.zeros(R, dtype=np.int32)
        
        # 각 행(회차)별로 실제 번호와 비교
        # 고급 인덱싱 or 반복문 사용 (NumPy 반복은 빠름)
        # 여기서는 정확성을 위해 간단한 row-wise loop 사용 (여전히 Python dict보다 훨씬 빠름)
        for r in range(R):
            # top6_indices[r] contains 0-based indices
            # actuals[r] is 0/1 array
            row_hits = 0
            for idx in top6_indices[r]:
                if actuals[r, idx] == 1:
                    row_hits += 1
            hits[r] = row_hits
            
        # 통계 집계
        hit_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        unique, counts = np.unique(hits, return_counts=True)
        for u, c in zip(unique, counts):
            hit_counts[int(u)] = int(c)
            
        return float(np.mean(hits)), hit_counts
