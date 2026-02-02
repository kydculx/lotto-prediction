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
            
        for num in range(1, 46):
            # 1. 시계열 생성 (출현: 1, 미출현: -1 로 중심화)
            series = np.zeros(n_draws)
            for i, row in enumerate(self.numbers_matrix):
                if num in row:
                    series[i] = 1.0
                else:
                    series[i] = -1.0 # 평균을 0 근처로 맞추기 위해 -1 사용
            
            # 2. FFT 수행
            fft_result = np.fft.fft(series)
            
            # 3. 고주파 노이즈 제거 (반등 위주 분석을 위해 저주파/유의미한 파장 강조)
            # 신호의 뒷부분(고주파)을 일부 감쇠
            # fft_result[len(fft_result)//2:] = 0
            
            # 4. 역푸리에 변환을 통해 '다음 시점'의 값을 외삽(Extrapolation)
            # FFT의 성질을 이용해 원래 신호를 복원하되, 다음 인덱스(n_draws)에서의 값을 구함
            # x[n] = sum(X[k] * exp(j * 2pi * k * n / N)) / N
            n = n_draws # 다음 회차 인덱스
            N = n_draws
            
            # 주파수 성분 결합
            k = np.arange(N)
            # 다음 점에 대한 복소평면상의 회전 기여도 계산
            extrapolated_val = np.sum(fft_result * np.exp(1j * 2 * np.pi * k * n / N)) / N
            
            # 실수부 추출 (실제 신호의 연장선)
            score_val = float(np.real(extrapolated_val))
            
            # 5. 점수 정규화 (보통 -1.0 ~ 1.0 사이로 나옴)
            # 시그모이드 형태나 단순 선형 정규화
            norm_score = (score_val + 1.0) / 2.0
            scores[num] = max(0.0, min(1.0, norm_score))
            
        # 점수 편차가 작을 경우 변별력을 위해 스케일링
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
