"""
번호 간격(Gap) 분석 엔진
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from .base import BaseEngine


class GapEngine(BaseEngine):
    """번호 간격 분석 엔진 (끝자리/처음 번호 분포 기반)"""
    
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
        scores = self.get_scores()
        return sorted([num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_numbers]])
