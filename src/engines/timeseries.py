"""
시계열 분석 엔진
"""

import numpy as np
from typing import Dict, List, Tuple
from .base import BaseEngine


class TimeSeriesEngine(BaseEngine):
    """시계열 분석 엔진"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        super().__init__(numbers_matrix)
        self.binary_matrix = self._create_binary_matrix()
        
    def _create_binary_matrix(self) -> np.ndarray:
        """이진 매트릭스 변환 (회차 x 45)"""
        binary = np.zeros((self.n_draws, 45), dtype=np.int8)
        for i, row in enumerate(self.numbers_matrix):
            for num in row:
                binary[i, num - 1] = 1
        return binary
    
    def get_moving_average(self, window: int = 20) -> Dict[int, np.ndarray]:
        """이동 평균 출현 빈도"""
        result = {}
        for num in range(1, 46):
            series = self.binary_matrix[:, num - 1]
            ma = np.convolve(series, np.ones(window)/window, mode='valid')
            result[num] = ma
        return result
    
    def get_trend(self, window: int = 30) -> Dict[int, str]:
        """최근 추세 분석"""
        ma = self.get_moving_average(window)
        trend = {}
        for num in range(1, 46):
            series = ma[num]
            if len(series) < 2:
                trend[num] = 'stable'
                continue
            recent, past = np.mean(series[-10:]), np.mean(series[-30:-10]) if len(series) >= 30 else np.mean(series[:-10])
            diff = recent - past
            trend[num] = 'rising' if diff > 0.02 else ('falling' if diff < -0.02 else 'stable')
        return trend
    
    def detect_periodicity(self, num: int) -> Dict:
        """출현 주기 분석"""
        series = self.binary_matrix[:, num - 1]
        appearances = np.where(series == 1)[0]
        if len(appearances) < 3: return {'avg_period': None, 'regular': False, 'next_expected': None, 'overdue': self.n_draws}
        
        gaps = np.diff(appearances)
        avg, std = np.mean(gaps), np.std(gaps)
        regularity = std / avg if avg > 0 else float('inf')
        return {
            'avg_period': avg, 'regular': regularity < 0.5, 'regularity_score': 1 - min(regularity, 1),
            'next_expected': appearances[-1] + int(avg), 'overdue': self.n_draws - 1 - appearances[-1]
        }
    
    def get_momentum(self, short_window: int = 10, long_window: int = 30) -> Dict[int, float]:
        """모멘텀 지표"""
        short_ma, long_ma = self.get_moving_average(short_window), self.get_moving_average(long_window)
        momentum = {}
        for num in range(1, 46):
            s, l = short_ma[num], long_ma[num]
            min_len = min(len(s), len(l))
            momentum[num] = float(s[-1] - l[-1]) if min_len > 0 else 0.0
        return momentum
    
    def get_scores(self) -> Dict[int, float]:
        """시계열 기반 점수 계산"""
        scores = {}
        trends = self.get_trend()
        trend_map = {'rising': 1.0, 'stable': 0.5, 'falling': 0.2}
        momentum = self.get_momentum()
        max_mom = max(abs(v) for v in momentum.values()) if momentum.values() else 1
        
        for num in range(1, 46):
            period_info = self.detect_periodicity(num)
            period_score = min(period_info.get('overdue', 0) / (period_info.get('avg_period') or 10), 2.0) / 2.0
            mom_score = (momentum[num] / max_mom + 1) / 2
            scores[num] = trend_map.get(trends[num], 0.5) * 0.3 + period_score * 0.4 + mom_score * 0.3
        return scores
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        """시계열 기반 예측"""
        scores = self.get_scores()
        return sorted([num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_numbers]])
