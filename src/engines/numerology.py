"""
숫자론 분석 엔진
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from .base import BaseEngine


class NumerologyEngine(BaseEngine):
    """숫자론 기반 분석 엔진"""
    
    PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
    SQUARES = {1, 4, 9, 16, 25, 36}
    FIBONACCI = {1, 2, 3, 5, 8, 13, 21, 34}
    
    def analyze_sum(self) -> Dict:
        sums = [sum(row) for row in self.numbers_matrix]
        return {
            'mean': np.mean(sums), 'std': np.std(sums),
            'optimal_range': (int(np.mean(sums) - np.std(sums)), int(np.mean(sums) + np.std(sums)))
        }
    
    def analyze_prime_ratio(self) -> Dict:
        prime_counts = [sum(1 for n in row if n in self.PRIMES) for row in self.numbers_matrix]
        counter = Counter(prime_counts)
        return {'optimal_count': counter.most_common(1)[0][0] if counter else 2}
    
    def analyze_digit_sum(self) -> Dict:
        digit_sum_counts = Counter()
        for row in self.numbers_matrix:
            for n in row: digit_sum_counts[sum(int(d) for d in str(n))] += 1
        total = sum(digit_sum_counts.values()) or 1
        return {'distribution': {k: v/total for k, v in digit_sum_counts.items()}}
    
    def get_scores(self) -> Dict[int, float]:
        scores = {}
        prime_opt = self.analyze_prime_ratio()['optimal_count']
        digit_sum_dist = self.analyze_digit_sum()['distribution']
        recent_avg_sum = np.mean([sum(row) for row in self.numbers_matrix[-30:]])
        
        for num in range(1, 46):
            score = (0.25 if num in self.PRIMES and prime_opt >= 2 else 0.15)
            score += (0.15 if num in self.SQUARES else 0.1)
            score += digit_sum_dist.get(sum(int(d) for d in str(num)), 0.05) * 3
            score += max(0, (1 - abs(num - recent_avg_sum/6)/20)) * 0.3
            scores[num] = min(score, 1.0)
        return scores
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        scores = self.get_scores()
        sum_analysis = self.analyze_sum()
        target_sum, sum_range = int(sum_analysis['mean']), sum_analysis['optimal_range']
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_selection = [num for num, _ in sorted_nums[:n_numbers]]
        # 심플한 합계 조정 로직 (리팩토링용)
        if sum(best_selection) < sum_range[0] or sum(best_selection) > sum_range[1]:
            # 조정 로직 생략 (기본 점수 기반 반환)
            pass
        return sorted(best_selection)
