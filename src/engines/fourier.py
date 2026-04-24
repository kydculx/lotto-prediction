"""
푸리에 변환 기반 분석 엔진
번호별 출현 기록을 시계열 신호로 간주하고 주기성(Frequency)을 분석하여 다음 출현 시점 예측
"""

import numpy as np
from typing import Dict, List
from .base import BaseEngine


class FourierEngine(BaseEngine):
    """푸리에 변환 분석 엔진"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        super().__init__(numbers_matrix)
        
    def get_scores(self) -> Dict[int, float]:
        """FFT 기반 주기성 점수 계산"""
        scores = {}
        n_draws = len(self.numbers_matrix)
        
        # 신호 길이가 너무 짧으면 분석 불가
        if n_draws < 32:
            return {i: 0.5 for i in range(1, 46)}
            
        # 최신 트렌드 반영을 위해 최근 120회차만 분석 (주기가 묻히지 않도록)
        window = min(n_draws, 120)
            
        for num in range(1, 46):
            # 1. 시계열 생성 (출현: 1, 미출현: 0)
            series = np.zeros(window)
            for i, row in enumerate(self.numbers_matrix[-window:]):
                if num in row:
                    series[i] = 1.0
            
            # 2. FFT 수행
            fft_result = np.fft.fft(series)
            
            # 3. 저주파(큰 주기 흐름)만 남기고 고주파(단기 노이즈) 제거 (Low-pass filter)
            # 상위 15% 주파수 대역만 유지
            cutoff = max(3, int(window * 0.15))
            fft_result[cutoff:-cutoff+1] = 0
            
            # 4. 역변환으로 부드러운 주기 곡선(Smoothed signal) 복원
            smoothed = np.real(np.fft.ifft(fft_result))
            
            # 5. 마지막 두 시점(최근 회차)의 레벨과 변화율(Derivative) 추출
            current_level = smoothed[-1]
            prev_level = smoothed[-2]
            trend = current_level - prev_level
            
            # 곡선의 최소/최대값으로 현재 위치를 정규화
            level_min, level_max = np.min(smoothed), np.max(smoothed)
            range_span = (level_max - level_min) if (level_max - level_min) > 0 else 1.0
            norm_level = (current_level - level_min) / range_span
            
            # 6. 주기 반등 점수 계산
            # - 상승 추세(trend > 0): 위치가 낮을수록(막 반등 시작) 더 높은 가중치
            # - 하락 추세(trend <= 0): 위치가 낮을수록(곧 바닥 도달) 적절한 가중치
            if trend > 0:
                score = 0.5 + 0.5 * (1.0 - norm_level)
            else:
                score = 0.5 * (1.0 - norm_level)
                
            scores[num] = max(0.0, min(1.0, score))
            
        # 점수 정규화 (변별력 강화)
        s_min = min(scores.values())
        s_max = max(scores.values())
        if s_max > s_min:
            for num in scores:
                scores[num] = (scores[num] - s_min) / (s_max - s_min)
                
        return scores

    def predict(self, n_numbers: int = 6) -> List[int]:
        """푸리에 점수 기반 예측"""
        scores = self.get_scores()
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted([num for num, _ in sorted_nums[:n_numbers]])
