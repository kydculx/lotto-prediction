"""
고급 패턴 분석 엔진
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from .base import BaseEngine


class AdvancedPatternEngine(BaseEngine):
    """고급 패턴 분석 엔진"""
    
    def analyze_markov_transitions(self) -> Dict[int, Dict[int, float]]:
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(self.n_draws - 1):
            curr_nums, next_nums = set(self.numbers_matrix[i]), set(self.numbers_matrix[i + 1])
            for curr in curr_nums:
                for nxt in next_nums: transitions[curr][nxt] += 1
        
        probs = {}
        for curr, next_counts in transitions.items():
            total = sum(next_counts.values()) or 1
            probs[curr] = {nxt: cnt/total for nxt, cnt in next_counts.items()}
        return probs
    
    def analyze_skip_patterns(self) -> Dict[int, float]:
        skip_history = defaultdict(list)
        for num in range(1, 46):
            skip = 0
            for row in self.numbers_matrix:
                if num in row: skip_history[num].append(skip); skip = 0
                else: skip += 1
        
        current_skips = {}
        for num in range(1, 46):
            s = 0
            for row in reversed(self.numbers_matrix):
                if num in row: break
                s += 1
            current_skips[num] = s
            
        probs = {}
        for num in range(1, 46):
            skips = skip_history[num]
            probs[num] = sum(1 for s in skips if s >= current_skips[num]) / len(skips) if skips else 0.5
        return probs
    
    def get_scores(self) -> Dict[int, float]:
        scores = {i: 0.0 for i in range(1, 46)}
        last_draw = set(self.numbers_matrix[-1])
        transitions = self.analyze_markov_transitions()
        skips = self.analyze_skip_patterns()
        
        markov_scores = {i: 0.0 for i in range(1, 46)}
        for num in last_draw:
            if num in transitions:
                for nxt, p in transitions[num].items(): markov_scores[nxt] += p
        max_m = max(markov_scores.values()) or 1
        
        for num in range(1, 46):
            scores[num] = (markov_scores[num]/max_m) * 0.5 + skips[num] * 0.5
        return scores
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        scores = self.get_scores()
        return sorted([num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_numbers]])
