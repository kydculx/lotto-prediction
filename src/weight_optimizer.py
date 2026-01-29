"""
가중치 자동 최적화기
백테스팅 결과를 기반으로 각 엔진의 가중치를 최적화
"""

import numpy as np
from typing import Dict, List, Tuple
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class WeightOptimizer:
    """엔진 가중치 최적화기"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        self.numbers_matrix = numbers_matrix
        self.n_draws = len(numbers_matrix)
        self.engine_names = [
            'statistical', 'pattern', 'timeseries', 'lstm',
            'graph', 'numerology', 'advanced_pattern', 'sequence_correlation',
            'gap'  # Gap 엔진 추가
        ]
        
    def _create_engines(self, train_matrix: np.ndarray) -> Dict:
        """EnsemblePredictor를 활용하여 엔진 생성"""
        from src.ensemble_predictor import EnsemblePredictor
        predictor = EnsemblePredictor(train_matrix, use_ml=False, use_validator=False)
        return predictor.engines
    
    def _get_ensemble_scores(self, engines: Dict, weights: Dict[str, float]) -> Dict[int, float]:
        """가중 앙상블 점수"""
        ensemble = {i: 0.0 for i in range(1, 46)}
        
        for name, engine in engines.items():
            try:
                scores = engine.get_scores()
                weight = weights.get(name, 0)
                for num, score in scores.items():
                    ensemble[num] += score * weight
            except:
                pass
        
        max_score = max(ensemble.values()) if ensemble.values() else 1
        if max_score > 0:
            ensemble = {k: v / max_score for k, v in ensemble.items()}
        
        return ensemble
    
    def _predict_with_weights(self, engines: Dict, weights: Dict[str, float]) -> List[int]:
        """가중치로 예측"""
        scores = self._get_ensemble_scores(engines, weights)
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:6]]
    
    def _evaluate_weights(self, weights: Dict[str, float], 
                         test_rounds: int = 50) -> Tuple[float, Dict]:
        """
        가중치 평가
        
        Returns:
            (평균 적중 수, 상세 결과)
        """
        if self.n_draws < test_rounds + 100:
            return 0.0, {}
        
        hit_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        
        for i in range(test_rounds):
            test_idx = self.n_draws - test_rounds + i
            train_matrix = self.numbers_matrix[:test_idx]
            
            if len(train_matrix) < 100:
                continue
            
            # 엔진 생성 및 예측
            engines = self._create_engines(train_matrix)
            predicted = self._predict_with_weights(engines, weights)
            
            # 실제 번호
            actual = set(self.numbers_matrix[test_idx])
            
            hits = len(set(predicted) & actual)
            hit_counts[hits] += 1
        
        total_tests = sum(hit_counts.values())
        avg_hits = sum(h * c for h, c in hit_counts.items()) / total_tests if total_tests > 0 else 0
        
        return avg_hits, hit_counts
    
    def grid_search(self, n_steps: int = 5, test_rounds: int = 30) -> Tuple[Dict[str, float], float]:
        """
        그리드 서치로 최적 가중치 탐색
        
        Args:
            n_steps: 각 가중치의 단계 수
            test_rounds: 테스트 회차 수
            
        Returns:
            (최적 가중치, 최고 점수)
        """
        print("⏳ 가중치 최적화 시작...")
        
        # 간소화된 그리드 (주요 엔진만)
        main_engines = ['statistical', 'advanced_pattern', 'sequence_correlation']
        other_engines = [n for n in self.engine_names if n not in main_engines]
        
        best_weights = None
        best_score = 0
        
        # 주요 엔진 가중치 탐색
        weight_options = [i / n_steps for i in range(1, n_steps)]
        
        tested = 0
        for w1 in weight_options:
            for w2 in weight_options:
                for w3 in weight_options:
                    # 나머지 가중치 분배
                    remaining = max(0, 1 - w1 - w2 - w3)
                    other_weight = remaining / len(other_engines) if other_engines else 0
                    
                    weights = {
                        main_engines[0]: w1,
                        main_engines[1]: w2,
                        main_engines[2]: w3,
                    }
                    for eng in other_engines:
                        weights[eng] = other_weight
                    
                    # 평가
                    score, _ = self._evaluate_weights(weights, test_rounds)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = weights.copy()
                    
                    tested += 1
                    if tested % 20 == 0:
                        print(f"  진행: {tested}개 조합 테스트, 현재 최고: {best_score:.3f}")
        
        print(f"✅ 최적화 완료! 최고 점수: {best_score:.3f}")
        
        return best_weights, best_score
    
    def quick_optimize(self, test_rounds: int = 50) -> Dict[str, float]:
        """
        빠른 최적화 (기본 가중치에서 미세 조정)
        """
        # 기본 가중치
        base_weights = {
            'statistical': 0.12,
            'pattern': 0.08,
            'timeseries': 0.08,
            'lstm': 0.10,
            'graph': 0.06,
            'numerology': 0.06,
            'advanced_pattern': 0.18,
            'sequence_correlation': 0.15,
            'gap': 0.08,
        }
        
        best_weights = base_weights.copy()
        best_score, _ = self._evaluate_weights(base_weights, test_rounds)
        
        print(f"기본 가중치 점수: {best_score:.3f}")
        
        # 각 엔진 가중치 미세 조정
        adjustments = [-0.05, 0.05]
        
        for engine in self.engine_names:
            for adj in adjustments:
                test_weights = best_weights.copy()
                test_weights[engine] = max(0.01, min(0.4, test_weights[engine] + adj))
                
                # 정규화
                total = sum(test_weights.values())
                test_weights = {k: v / total for k, v in test_weights.items()}
                
                score, _ = self._evaluate_weights(test_weights, test_rounds)
                
                if score > best_score:
                    best_score = score
                    best_weights = test_weights.copy()
                    print(f"  {engine} 조정 후: {best_score:.3f}")
        
        return best_weights


# 테스트
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace('/src/weight_optimizer.py', ''))
    from src.data_loader import LottoDataLoader
    
    loader = LottoDataLoader()
    loader.load()
    matrix = loader.get_numbers_matrix()
    
    optimizer = WeightOptimizer(matrix)
    
    print("=" * 50)
    print("🔧 가중치 최적화 테스트")
    print("=" * 50)
    
    optimized = optimizer.quick_optimize(test_rounds=30)
    
    print("\n📊 최적화된 가중치:")
    for name, weight in sorted(optimized.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {weight:.3f}")
