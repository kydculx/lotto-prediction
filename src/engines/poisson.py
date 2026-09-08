"""
포아송 분포 기반 분석 엔진
최근 출현 빈도가 기대값보다 낮은 번호들을 '반등 가능성' 점수로 계산
"""

import numpy as np
from math import exp, log, lgamma
from typing import Dict, List
from .base import BaseEngine


class PoissonEngine(BaseEngine):
    """포아송 분포 분석 엔진"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        super().__init__(numbers_matrix)

    def _poisson_cdf(self, k: int, mu: float) -> float:
        """포아송 누적 분포 — log-gamma로 수치 안정성 확보"""
        if mu <= 0:
            return 0.0
        # log(P(X=i)) = i*log(mu) - mu - log(i!)
        # CDF = sum(P(X=i) for i in 0..k), 누적 합산은 별도 PMF 호출 없이
        # 직접 구현보다 정규화된 gamma 불완전 함수를 사용하는 것이 안정적이나,
        # mu가 작고 k가 작은 범위(<50)이므로 log-gamma로 계산된 PMF를 누적
        pmf = 0.0
        cdf = 0.0
        log_mu = log(mu)
        for i in range(k + 1):
            if i == 0:
                pmf = exp(-mu)
            else:
                # PMF 재귀: P(X=i) = P(X=i-1) * mu / i (수치적으로 안정)
                pmf = pmf * mu / i
            cdf += pmf
        return min(cdf, 1.0)

    def get_scores(self) -> Dict[int, float]:
        """포아송 기반 반등 가능성 점수 계산"""
        scores = {}
        
        # 전체 평균 출현 확률 (약 6/45 = 0.1333)
        # 하지만 번호별로 편차가 있을 수 있으므로 실제 전체 이력을 기반으로 계산
        total_draws = len(self.numbers_matrix)
        all_flatten = self.numbers_matrix.flatten()
        counts = np.bincount(all_flatten, minlength=46)
        
        # 윈도우 사이즈 (최근 50회차)
        window_size = min(50, total_draws)
        recent_data = self.numbers_matrix[-window_size:].flatten()
        recent_counts = np.bincount(recent_data, minlength=46)
        
        for num in range(1, 46):
            # 1. 장기 기대 확률
            p_expected = counts[num] / total_draws if total_draws > 0 else 6/45
            
            # 2. 최근 윈도우에서의 기대값 (mu)
            mu = p_expected * window_size
            
            # 3. 최근 실제 출현 횟수 (k)
            k = recent_counts[num]
            
            # 4. 점수화: 실제 출현이 기대보다 낮을수록(과소평가) 높은 점수
            # P(K <= k) 가 작을수록 '운이 나쁜' 상태 -> 반등 기대
            # 만약 mu가 너무 작으면(신규 데이터 등) 기본값 처리
            if mu > 0:
                prob_less_than_actual = self._poisson_cdf(k, mu)
                # 점수 = 1.0 - 누적확률 (확률이 낮을수록 1.0에 수렴)
                # 즉, 발생하기 힘든 낮은 빈도일수록 점수가 높음
                score = 1.0 - prob_less_than_actual
                
                # 가중치 조정: 너무 극단적인 점수 방지
                score = max(0.1, min(0.9, score))
                
                # 추가 보너스: 최근 10회차 연속 미출현시 가중
                last_10 = self.numbers_matrix[-10:].flatten()
                if num not in last_10:
                    score += 0.1
            else:
                score = 0.5
                
            scores[num] = float(score)
            
        return scores

    def predict(self, n_numbers: int = 6) -> List[int]:
        """포아송 점수 기반 예측"""
        scores = self.get_scores()
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted([num for num, _ in sorted_nums[:n_numbers]])
