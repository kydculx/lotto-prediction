"""
그래프 이론 분석 엔진
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from itertools import combinations
from .base import BaseEngine


class GraphEngine(BaseEngine):
    """그래프 이론 기반 번호 관계 분석 엔진"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        super().__init__(numbers_matrix)
        self.cooccurrence_matrix = self._build_cooccurrence_matrix()
        
    def _build_cooccurrence_matrix(self) -> np.ndarray:
        matrix = np.zeros((45, 45), dtype=np.int32)
        for row in self.numbers_matrix:
            for i, j in combinations(row, 2):
                matrix[i-1, j-1] += 1
                matrix[j-1, i-1] += 1
        return matrix
    
    def get_number_partners(self, num: int, top_k: int = 5) -> List[Tuple[int, int]]:
        idx = num - 1
        partners = [(i+1, self.cooccurrence_matrix[idx, i]) for i in range(45) if i != idx]
        return sorted(partners, key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_centrality(self) -> Dict[int, float]:
        centrality = {}
        for num in range(1, 46):
            idx = num - 1
            total, n_connected = np.sum(self.cooccurrence_matrix[idx, :]), np.sum(self.cooccurrence_matrix[idx, :] > 0)
            centrality[num] = total * (n_connected / 44)
        max_cent = max(centrality.values()) or 1
        return {k: v / max_cent for k, v in centrality.items()}
    
    def get_scores(self) -> Dict[int, float]:
        scores = {}
        centrality = self.get_centrality()
        recent_freq = Counter(self.numbers_matrix[-30:].flatten())
        
        partner_scores = {}
        for num in range(1, 46):
            partners = self.get_number_partners(num, top_k=10)
            partner_scores[num] = sum(recent_freq[p] * cnt for p, cnt in partners)
        max_p = max(partner_scores.values()) or 1
        
        for num in range(1, 46):
            scores[num] = centrality[num] * 0.5 + (partner_scores[num] / max_p) * 0.5
        return scores
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        scores = self.get_scores()
        return sorted([num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_numbers]])
