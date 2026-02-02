"""
앙상블 예측기 v3.0
9개 분석 엔진 + ML 모델 + 조합 검증 + 동적 가중치 최적화
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from itertools import combinations


class EnsemblePredictor:
    """앙상블 예측기 v3.0"""
    
    # 최적화된 엔진 가중치 (자동 최적화)
    DEFAULT_WEIGHTS = {
        'poisson': 0.2396,
        'advanced_pattern': 0.2172,
        'statistical': 0.2000,
        'gap': 0.0800,
        'graph': 0.0745,
        'timeseries': 0.0599,
        'fourier': 0.0555,
        'pattern': 0.0501,
        'ml': 0.0200,                # 추가
        'lstm': 0.0168,
        'sequence_correlation': 0.0063,
        'numerology': 0.0001,        # 추가
    }
    
    def __init__(self, numbers_matrix: np.ndarray, 
                 weights: Dict[str, float] = None,
                 use_ml: bool = True,
                 use_validator: bool = True,
                 use_dynamic_weight: bool = True): # 동적 가중치 옵션 추가
        self.numbers_matrix = numbers_matrix
        self.use_ml = use_ml
        self.use_validator = use_validator
        self.use_dynamic_weight = use_dynamic_weight
        
        self.engines = {}
        self.engine_scores = {}
        self.engine_predictions = {}
        self.dynamic_boosts = {} # 엔진별 성능 가중치 부스트
        self.validator = None
        self.optimizer = None
        
        # 엔진 동적 로드
        self._load_engines()
        
        # 가중치 설정 (로드된 엔진 기준)
        self.base_weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        if self.use_dynamic_weight:
            self._calculate_dynamic_boosts()
            
        self._normalize_weights()
        
        self._analyze_sum_stats()
        self._initialize_validator()
        
    def _load_engines(self):
        """src.engines 패키지에서 엔진들을 동적으로 로드"""
        import importlib
        import pkgutil
        import src.engines as engines_pkg
        from src.engines.base import BaseEngine
        
        for loader, module_name, is_pkg in pkgutil.walk_packages(engines_pkg.__path__, engines_pkg.__name__ + "."):
            if module_name == 'src.engines.base':
                continue
                
            module = importlib.import_module(module_name)
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, BaseEngine) and obj is not BaseEngine:
                    # 엔진 인스턴스화
                    engine_id = obj.__name__.replace('Engine', '').lower()
                    
                    # ML 엔진 제외 처리 (use_ml=False일 때)
                    if not self.use_ml and engine_id == 'ml':
                        continue
                        
                    try:
                        instance = obj(self.numbers_matrix)
                        # ML 엔진은 추가 학습 필요
                        if engine_id == 'ml':
                            if not instance.train():
                                continue
                                
                        self.engines[engine_id] = instance
                    except Exception as e:
                        print(f"⚠️ 엔진 {engine_id} 초기화 실패: {e}")

    def _calculate_dynamic_boosts(self):
        """최근 10회차 엔진별 성능을 기반으로 가중치 부스트 계산 (메타 러닝)"""
        lookback = 10
        if len(self.numbers_matrix) < lookback + 50:
            self.dynamic_boosts = {k: 1.0 for k in self.engines}
            return

        performance = {k: 0.0 for k in self.engines}
        
        # 최근 lookback 회차 동안 각 엔진의 적중 내역 확인
        for i in range(1, lookback + 1):
            idx = -i
            train_matrix = self.numbers_matrix[:idx]
            actual = set(self.numbers_matrix[idx])
            
            for name, engine_class in self.engines.items():
                try:
                    # 임시 엔진 생성 (현재 idx까지의 데이터로)
                    # ML 엔진은 너무 느리므로 성능 최적화를 위해 일부 엔진만 정밀 검증하거나
                    # 기존 예측 데이터를 캐싱하는 방식이 좋으나, 여기선 단순화
                    if name == 'ml' or name == 'lstm':
                        # 무거운 엔진은 계산 건너뛰거나 기본값 유지
                        continue
                        
                    temp_engine = self.engines[name].__class__(train_matrix)
                    pred = set(temp_engine.predict())
                    hits = len(pred & actual)
                    performance[name] += hits
                except:
                    continue
        
        # 부스트 계산 (평균 적중수 기반, 최소 0.8 ~ 최대 1.3)
        max_perf = max(performance.values()) if any(performance.values()) else 1
        for name in self.engines:
            if max_perf > 0:
                # 성능이 좋을수록 부스트 (최대 30% 증가)
                boost = 1.0 + (performance[name] / max_perf) * 0.3
            else:
                boost = 1.0
            self.dynamic_boosts[name] = boost

    def _normalize_weights(self):
        """현재 로드된 엔진들과 동적 부스트를 반영하여 가중치 정규화"""
        temp_weights = {}
        for k in self.engines:
            base = self.base_weights.get(k, 0.05)
            boost = self.dynamic_boosts.get(k, 1.0)
            temp_weights[k] = base * boost
            
        total = sum(temp_weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in temp_weights.items()}
        else:
            w = 1.0 / len(self.engines) if self.engines else 1.0
            self.weights = {k: w for k in self.engines}

    def _analyze_sum_stats(self):
        """합계 통계 분석"""
        sums = [sum(row) for row in self.numbers_matrix]
        self.mean_sum = np.mean(sums) if sums else 138
        self.std_sum = np.std(sums) if sums else 20
        self.min_optimal_sum = self.mean_sum - self.std_sum
        self.max_optimal_sum = self.mean_sum + self.std_sum
        
    def _initialize_validator(self):
        """조합 검증기 초기화"""
        if self.use_validator:
            try:
                from src.combination_validator import CombinationValidator, CombinationOptimizer
                self.validator = CombinationValidator()
                self.optimizer = CombinationOptimizer(self.numbers_matrix)
            except Exception as e:
                print(f"⚠️ 조합 검증기 초기화 실패: {e}")
                self.use_validator = False
    
    def calculate_all_scores(self) -> Dict[str, Dict[int, float]]:
        """모든 엔진의 점수 계산"""
        self.engine_scores = {}
        
        for name, engine in self.engines.items():
            try:
                self.engine_scores[name] = engine.get_scores()
            except Exception as e:
                self.engine_scores[name] = {i: 0.5 for i in range(1, 46)}
                
        return self.engine_scores
    
    def get_all_predictions(self) -> Dict[str, List[int]]:
        """모든 엔진의 예측 결과"""
        self.engine_predictions = {}
        
        for name, engine in self.engines.items():
            try:
                self.engine_predictions[name] = engine.predict()
            except Exception as e:
                self.engine_predictions[name] = []
                
        return self.engine_predictions
    
    def get_ensemble_scores(self) -> Dict[int, float]:
        """가중 평균 앙상블 점수 + 투표 기반 부스트"""
        if not self.engine_scores:
            self.calculate_all_scores()
        if not self.engine_predictions:
            self.get_all_predictions()
            
        ensemble = {i: 0.0 for i in range(1, 46)}
        
        # 1. 가중 평균 점수 (55%)
        total_weight = sum(self.weights.get(name, 0) for name in self.engines.keys())
        for name, scores in self.engine_scores.items():
            weight = self.weights.get(name, 0) / total_weight if total_weight > 0 else 0
            for num, score in scores.items():
                ensemble[num] += score * weight * 0.55
        
        # 2. 투표 기반 점수 (30%)
        vote_counts = Counter()
        for predictions in self.engine_predictions.values():
            for num in predictions:
                vote_counts[num] += 1
        
        max_votes = max(vote_counts.values()) if vote_counts.values() else 1
        for num in range(1, 46):
            vote_score = vote_counts.get(num, 0) / max_votes
            ensemble[num] += vote_score * 0.30
        
        # 3. 직전 회차 반복 보너스 (15%)
        last_draw = set(self.numbers_matrix[-1])
        for num in last_draw:
            ensemble[num] += 0.15
        
        # 정규화
        max_score = max(ensemble.values()) if ensemble.values() else 1
        if max_score > 0:
            ensemble = {k: v / max_score for k, v in ensemble.items()}
            
        return ensemble
    
    def _optimize_combination(self, candidates: List[Tuple[int, float]], 
                              n_numbers: int = 6) -> List[int]:
        """
        조합 최적화 (AC값, 홀짝, 연속번호 등 고려)
        """
        top_n = min(20, len(candidates))
        top_candidates = [num for num, _ in candidates[:top_n]]
        
        best_combo = None
        best_score = -float('inf')
        
        # 점수 딕셔너리 미리 생성 (최적화)
        scores_dict = dict(candidates)
        
        for combo in combinations(top_candidates, n_numbers):
            combo_list = list(combo)
            combo_sum = sum(combo)
            
            # 기본 조합 검증
            if self.validator:
                is_valid, results = self.validator.validate(combo_list)
                validator_score = self.validator.score(combo_list)
            else:
                is_valid = True
                validator_score = 0.5
            
            # 합계 적합도 (25%)
            if self.min_optimal_sum <= combo_sum <= self.max_optimal_sum:
                sum_score = 1.0
            else:
                distance = min(abs(combo_sum - self.min_optimal_sum), 
                             abs(combo_sum - self.max_optimal_sum))
                sum_score = max(0, 1 - distance / 40)
            
            # 번호 점수 (30%)
            scores_dict = dict(candidates)
            num_score = sum(scores_dict.get(n, 0) for n in combo) / n_numbers
            
            # 검증기 점수 (30%)
            valid_score = validator_score
            
            # 다양성 점수 (15%)
            sections_covered = len(set((n-1)//10 for n in combo))
            diversity_score = sections_covered / 5
            
            # 종합 점수
            total_score = (sum_score * 0.25 + num_score * 0.30 + 
                          valid_score * 0.30 + diversity_score * 0.15)
            
            if total_score > best_score:
                best_score = total_score
                best_combo = combo_list
        
        return sorted(best_combo) if best_combo else [num for num, _ in candidates[:n_numbers]]
    
    def calculate_confidence(self, numbers: List[int]) -> float:
        """예측 번호 조합의 신뢰도 계산"""
        if not self.engine_predictions:
            self.get_all_predictions()
            
        # 엔진 추천 횟수
        recommendation_counts = Counter()
        for predictions in self.engine_predictions.values():
            for num in predictions:
                recommendation_counts[num] += 1
        
        total_engines = len(self.engines)
        avg_recommendation = sum(recommendation_counts.get(n, 0) for n in numbers) / len(numbers)
        
        # 합계 적합도
        combo_sum = sum(numbers)
        if self.min_optimal_sum <= combo_sum <= self.max_optimal_sum:
            sum_confidence = 1.0
        else:
            distance = min(abs(combo_sum - self.min_optimal_sum), 
                         abs(combo_sum - self.max_optimal_sum))
            sum_confidence = max(0, 1 - distance / 50)
        
        # 조합 검증 점수
        if self.validator:
            validator_score = self.validator.score(numbers)
        else:
            validator_score = 0.5
        
        # 종합 신뢰도
        engine_confidence = (avg_recommendation / total_engines) * 100
        confidence = (engine_confidence * 0.4 + 
                     sum_confidence * 25 + 
                     validator_score * 35)
        
        return min(confidence, 100)
    
    def predict_single_set(self) -> Tuple[List[int], float]:
        """단일 예측 세트 생성"""
        ensemble_scores = self.get_ensemble_scores()
        sorted_nums = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = self._optimize_combination(sorted_nums)
        confidence = self.calculate_confidence(selected)
        
        return sorted(selected), confidence
    
    def predict_multiple_sets(self, n_sets: int = 5) -> List[Tuple[List[int], float]]:
        """다중 예측 세트 생성 (다양성 + 최적화)"""
        ensemble_scores = self.get_ensemble_scores()
        sorted_nums = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        used_combinations = set()
        
        # 첫 번째 세트: 최적화된 조합
        first_set = tuple(self._optimize_combination(sorted_nums))
        results.append((list(first_set), self.calculate_confidence(list(first_set))))
        used_combinations.add(first_set)
        
        # 직전 회차 반복 포함 세트
        last_draw = list(self.numbers_matrix[-1])
        top_scored = [num for num, _ in sorted_nums[:15]]
        
        for set_idx in range(1, n_sets):
            np.random.seed(set_idx * 42 + 7)
            
            if set_idx == 1:
                # 반복 번호 포함
                base = list(last_draw[:2])
                remaining = [n for n in top_scored if n not in base]
                np.random.shuffle(remaining)
                base.extend(remaining[:4])
                candidate = base[:6]
            else:
                # 다양성을 위한 변형
                top_20 = [num for num, _ in sorted_nums[:20]]
                weights = np.array([ensemble_scores[n] for n in top_20])
                weights = weights / weights.sum()
                
                attempts = 0
                candidate = None
                while attempts < 100:
                    selected_indices = np.random.choice(len(top_20), size=6, replace=False, p=weights)
                    candidate = sorted([top_20[i] for i in selected_indices])
                    
                    if tuple(candidate) not in used_combinations:
                        # 조합 검증
                        if self.validator:
                            is_valid, _ = self.validator.validate(candidate)
                            if is_valid or attempts > 50:
                                break
                        else:
                            break
                    attempts += 1
            
            if candidate and tuple(sorted(candidate)) not in used_combinations:
                candidate = sorted(candidate)
                used_combinations.add(tuple(candidate))
                results.append((candidate, self.calculate_confidence(candidate)))
        
        # 신뢰도 순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_hot_cold_analysis(self) -> Dict:
        """핫/콜드 넘버 요약"""
        if 'statistical' in self.engines:
            stat_engine = self.engines['statistical']
            return {
                'hot': stat_engine.get_hot_numbers(last_n=50, top_k=10),
                'cold': stat_engine.get_cold_numbers(last_n=50, top_k=10),
                'overdue': stat_engine.get_overdue_numbers()[:10]
            }
        return {}
    
    def get_repeat_analysis(self) -> Dict:
        """반복 출현 분석"""
        if 'advanced_pattern' in self.engines:
            adv_engine = self.engines['advanced_pattern']
            return {
                'repeat_candidates': adv_engine.get_repeat_candidates(),
                'analysis': adv_engine.analyze_consecutive_appearance()
            }
        return {}
    
    def get_combination_analysis(self, numbers: List[int]) -> Dict:
        """조합 상세 분석"""
        if self.validator:
            is_valid, results = self.validator.validate(numbers)
            results['score'] = self.validator.score(numbers)
            return results
        return {'is_valid': True, 'score': 0.5}
    
    def get_detailed_report(self, n_sets: int = 5) -> Dict:
        """상세 분석 리포트"""
        self.calculate_all_scores()
        self.get_all_predictions()
        
        predicted_sets = self.predict_multiple_sets(n_sets)
        
        return {
            'engine_predictions': self.engine_predictions,
            'final_weights': self.weights,
            'dynamic_boosts': self.dynamic_boosts,
            'ensemble_scores': self.get_ensemble_scores(),
            'hot_cold': self.get_hot_cold_analysis(),
            'repeat_analysis': self.get_repeat_analysis(),
            'predicted_sets': predicted_sets,
            'sum_range': (int(self.min_optimal_sum), int(self.max_optimal_sum)),
            'top_set_analysis': self.get_combination_analysis(predicted_sets[0][0]) if predicted_sets else {}
        }


# 테스트
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace('/src/ensemble_predictor.py', ''))
    
    from src.data_loader import LottoDataLoader
    
    loader = LottoDataLoader()
    loader.load()
    matrix = loader.get_numbers_matrix()
    
    print("=" * 50)
    print("🎱 앙상블 예측기 v3.0 테스트")
    print("=" * 50)
    
    print("\n⏳ ML 엔진 학습 중...")
    predictor = EnsemblePredictor(matrix, use_ml=True, use_validator=True)
    
    print("\n📊 엔진별 예측:")
    predictions = predictor.get_all_predictions()
    for name, nums in predictions.items():
        print(f"  {name}: {nums}")
    
    print("\n🎯 앙상블 예측 (5세트):")
    sets = predictor.predict_multiple_sets(5)
    for i, (nums, conf) in enumerate(sets, 1):
        analysis = predictor.get_combination_analysis(nums)
        print(f"  SET {i}: {nums}")
        print(f"         신뢰도: {conf:.1f}%, AC:{analysis.get('ac', '?')}, 합:{sum(nums)}")
