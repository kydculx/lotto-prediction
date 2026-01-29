"""
통계적 빈도 분석 엔진
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from .base import BaseEngine


class StatisticalEngine(BaseEngine):
    """통계적 빈도 분석 엔진"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        super().__init__(numbers_matrix)
        self.all_numbers = numbers_matrix.flatten()
        
    def get_frequency(self, last_n: int = None) -> Dict[int, int]:
        """번호별 출현 빈도 계산"""
        if last_n:
            data = self.numbers_matrix[-last_n:].flatten()
        else:
            data = self.all_numbers
            
        freq = Counter(data)
        return {i: freq.get(i, 0) for i in range(1, 46)}
    
    def get_hot_numbers(self, last_n: int = 50, top_k: int = 10) -> List[Tuple[int, int]]:
        """핫 넘버 반환"""
        freq = self.get_frequency(last_n)
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_freq[:top_k]
    
    def get_cold_numbers(self, last_n: int = 50, top_k: int = 10) -> List[Tuple[int, int]]:
        """콜드 넘버 반환"""
        freq = self.get_frequency(last_n)
        sorted_freq = sorted(freq.items(), key=lambda x: x[1])
        return sorted_freq[:top_k]
    
    def get_appearance_gap(self) -> Dict[int, Dict]:
        """번호별 출현 간격 분석"""
        result = {}
        for num in range(1, 46):
            appearances = [i for i, row in enumerate(self.numbers_matrix) if num in row]
            
            if not appearances:
                result[num] = {'last_seen': self.n_draws, 'avg_gap': float('inf'), 'gaps': []}
                continue
                
            gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
            last_seen = self.n_draws - 1 - appearances[-1]
            
            result[num] = {
                'last_seen': last_seen,
                'avg_gap': np.mean(gaps) if gaps else 0,
                'gaps': gaps
            }
        return result
    
    def get_overdue_numbers(self, threshold: float = 1.5) -> List[Tuple[int, float]]:
        """과도 지연 번호 반환"""
        gaps = self.get_appearance_gap()
        overdue = []
        for num, data in gaps.items():
            if data['avg_gap'] > 0:
                delay_ratio = data['last_seen'] / data['avg_gap']
                if delay_ratio >= threshold:
                    overdue.append((num, delay_ratio))
        return sorted(overdue, key=lambda x: x[1], reverse=True)
    
    def get_scores(self) -> Dict[int, float]:
        """통계적 점수 계산"""
        scores = {}
        recent_freq = self.get_frequency(last_n=30)
        max_freq = max(recent_freq.values()) if recent_freq.values() else 1
        
        mid_freq = self.get_frequency(last_n=100)
        max_mid = max(mid_freq.values()) if mid_freq.values() else 1
        
        gaps = self.get_appearance_gap()
        
        for num in range(1, 46):
            freq_score = recent_freq[num] / max_freq * 0.35
            mid_score = mid_freq[num] / max_mid * 0.15
            
            gap_data = gaps[num]
            if gap_data['avg_gap'] > 0:
                delay_ratio = gap_data['last_seen'] / gap_data['avg_gap']
                if delay_ratio >= 2.0: delay_score = 1.0
                elif delay_ratio >= 1.5: delay_score = 0.8
                elif delay_ratio >= 1.0: delay_score = 0.5
                else: delay_score = delay_ratio * 0.3
            else:
                delay_score = 0.3
            
            scores[num] = freq_score + mid_score + delay_score * 0.50
            
        return scores
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        """통계 기반 예측"""
        scores = self.get_scores()
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted([num for num, _ in sorted_nums[:n_numbers]])
