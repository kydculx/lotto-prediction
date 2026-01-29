"""
분석 엔진 베이스 클래스
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any


class BaseEngine(ABC):
    """모든 로또 분석 엔진의 추상 베이스 클래스"""
    
    def __init__(self, numbers_matrix: np.ndarray):
        """
        Args:
            numbers_matrix: 당첨번호 2D 배열 (회차 x 6개 번호)
        """
        self.numbers_matrix = numbers_matrix
        self.n_draws = len(numbers_matrix)
        
    @abstractmethod
    def get_scores(self) -> Dict[int, float]:
        """
        각 번호(1~45)에 대한 다음 회차 출현 가능성 점수 계산
        Returns: {번호: 점수(0.0~1.0)}
        """
        pass
        
    @abstractmethod
    def predict(self, n_numbers: int = 6) -> List[int]:
        """
        이 엔진의 분석 결과에 따른 추천 번호 조합 생성
        Returns: [번호1, 번호2, ..., 번호n]
        """
        pass

    def get_name(self) -> str:
        """엔진의 영문 아이디 반환"""
        return self.__class__.__name__.replace('Engine', '').lower()
