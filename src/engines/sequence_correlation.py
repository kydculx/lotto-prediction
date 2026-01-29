"""
연속 회차 상관관계 분석 엔진
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from .base import BaseEngine


class SequenceCorrelationEngine(BaseEngine):
    """연속 회차 상관관계 분석"""
    
    def analyze_next_number_probability(self, lookback: int = 3) -> Dict[int, float]:
        recent_nums = set(self.numbers_matrix[-lookback:].flatten())
        next_counts = Counter()
        for i in range(lookback, self.n_draws - 1):
            window_nums = set(self.numbers_matrix[i-lookback:i].flatten())
            similarity = len(recent_nums & window_nums) / len(recent_nums | window_nums) if recent_nums | window_nums else 0
            if similarity > 0.3:
                for num in self.numbers_matrix[i]: next_counts[num] += similarity
        total = sum(next_counts.values()) or 1
        return {i: next_counts.get(i, 0) / total for i in range(1, 46)}
    
    def get_likely_followers(self) -> List[int]:
        follows = defaultdict(Counter)
        for i in range(self.n_draws - 1):
            for curr in self.numbers_matrix[i]:
                for nxt in self.numbers_matrix[i+1]: follows[curr][nxt] += 1
        
        last_draw, follower_scores = self.numbers_matrix[-1], Counter()
        for num in last_draw:
            for i, (f, _) in enumerate(follows[num].most_common(6)): follower_scores[f] += (6 - i)
        return [num for num, _ in follower_scores.most_common(10)]
    
    def get_scores(self) -> Dict[int, float]:
        scores = {i: 0.0 for i in range(1, 46)}
        probs = self.analyze_next_number_probability()
        max_p = max(probs.values()) or 1
        followers = self.get_likely_followers()
        follower_map = {num: (len(followers) - i) / len(followers) for i, num in enumerate(followers)}
        
        for num in range(1, 46):
            scores[num] = (probs.get(num, 0) / max_p) * 0.5 + follower_map.get(num, 0) * 0.5
        return scores
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        scores = self.get_scores()
        return sorted([num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_numbers]])
