"""
딥러닝 LSTM 엔진
"""

import numpy as np
from typing import Dict, List, Tuple
from .base import BaseEngine
import warnings
warnings.filterwarnings('ignore')


class LSTMEngine(BaseEngine):
    """딥러닝 LSTM 예측 엔진 (경량 버전 포함)"""
    
    def __init__(self, numbers_matrix: np.ndarray, sequence_length: int = 10):
        super().__init__(numbers_matrix)
        self.sequence_length = sequence_length
        self.model = None
        self.binary_matrix = self._create_binary_matrix()
        
    def _create_binary_matrix(self) -> np.ndarray:
        binary = np.zeros((self.n_draws, 45), dtype=np.float32)
        for i, row in enumerate(self.numbers_matrix):
            for num in row:
                binary[i, num - 1] = 1.0
        return binary
    
    def predict_probabilities(self) -> np.ndarray:
        """확률 예측 (경량 가중 이동 평균 방식)"""
        weights = np.exp(np.linspace(-1, 0, self.sequence_length))
        weights /= weights.sum()
        recent = self.binary_matrix[-self.sequence_length:]
        weighted_avg = np.average(recent, axis=0, weights=weights)
        overall_avg = np.mean(self.binary_matrix, axis=0)
        return 0.7 * weighted_avg + 0.3 * overall_avg
    
    def get_scores(self) -> Dict[int, float]:
        probs = self.predict_probabilities()
        return {i+1: float(probs[i]) for i in range(45)}
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        probs = self.predict_probabilities()
        return sorted([int(idx + 1) for idx in np.argsort(probs)[-n_numbers:]])


def create_lstm_engine(numbers_matrix: np.ndarray, use_tensorflow: bool = False):
    """LSTM 엔진 생성 (리팩토링 버전은 기본적으로 경량 버전 사용)"""
    return LSTMEngine(numbers_matrix)
