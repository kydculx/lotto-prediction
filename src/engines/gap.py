"""
번호 간격(Gap) 분석 엔진
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from .base import BaseEngine


class GapEngine(BaseEngine):
    """번호 간격 분석 엔진"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        super().__init__(numbers_matrix)
        self._analyze_stats()
        
    def _analyze_stats(self):
        all_gaps = []
        for row in self.numbers_matrix:
            sorted_row = sorted(row)
            all_gaps.extend([sorted_row[i+1] - sorted_row[i] for i in range(5)])
        self.freq = Counter(all_gaps)
        self.optimal_gaps = {g for g, _ in self.freq.most_common(10)}
        self.mean_gap = np.mean(all_gaps) if all_gaps else 7
        
    def get_scores(self) -> Dict[int, float]:
        scores = {i: 0.0 for i in range(1, 46)}
        first_dist = Counter(min(row) for row in self.numbers_matrix)
        last_dist = Counter(max(row) for row in self.numbers_matrix)
        total = self.n_draws
        
        for num in range(1, 46):
            scores[num] = (first_dist.get(num, 0)/total) * 0.4 + (last_dist.get(num, 0)/total) * 0.4
            scores[num] += (Counter(self.numbers_matrix.flatten()).get(num, 0)/total) * 0.2
        max_s = max(scores.values()) or 1
        return {k: v/max_s for k, v in scores.items()}
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        # 심플한 패턴 생성 (리팩토링용)
        first_num = sorted(Counter(min(row) for row in self.numbers_matrix).items(), key=lambda x: x[1], reverse=True)[0][0]
        step = int(self.mean_gap)
        res = [first_num + i*step for i in range(n_numbers)]
        res = [min(max(1, n), 45) for n in res]
        # 중복 제거 및 부족분 채우기
        res = sorted(list(set(res)))
        while len(res) < n_numbers:
            for n in range(1, 46):
                if n not in res: res.append(n); break
        return sorted(res[:n_numbers])
