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
        """어텐션(Attention) 기반 과거 시퀀스 패턴 매칭 (유사 LSTM)"""
        # 최근 시퀀스 추출
        L = min(self.sequence_length, self.n_draws - 1)
        if L < 2:
            return np.mean(self.binary_matrix, axis=0)
            
        # 최신 흐름 (Context)
        current_seq = self.binary_matrix[-L:].flatten()
        norm_curr = np.linalg.norm(current_seq)
        if norm_curr == 0:
            return np.mean(self.binary_matrix, axis=0)
            
        similarities = []
        next_draws = []
        
        # 과거 데이터를 슬라이딩 윈도우로 탐색하며 현재 흐름과 가장 유사한 구간을 찾음
        for i in range(self.n_draws - L):
            past_seq = self.binary_matrix[i:i+L].flatten()
            norm_past = np.linalg.norm(past_seq)
            
            if norm_past > 0:
                # 코사인 유사도 (Cosine Similarity)
                sim = np.dot(current_seq, past_seq) / (norm_curr * norm_past)
            else:
                sim = 0.0
                
            # 해당 과거 패턴 바로 다음에 나온 결과
            next_draw = self.binary_matrix[i+L]
            
            similarities.append(sim)
            next_draws.append(next_draw)
            
        if not similarities:
            return np.mean(self.binary_matrix, axis=0)
            
        similarities = np.array(similarities)
        next_draws = np.array(next_draws)
        
        # 상위 K개의 가장 유사했던 과거 패턴 추출
        K = min(15, len(similarities))
        top_k_indices = np.argsort(similarities)[-K:]
        
        top_sims = similarities[top_k_indices]
        top_next_draws = next_draws[top_k_indices]
        
        # 유사도를 가중치로 사용하여 과거의 "다음 결과"들을 결합 (Attention Mechanism)
        weight_sum = np.sum(top_sims)
        if weight_sum > 0:
            weighted_pred = np.sum(top_next_draws * top_sims[:, np.newaxis], axis=0) / weight_sum
        else:
            weighted_pred = np.mean(self.binary_matrix, axis=0)
            
        # 전체 통계와 앙상블하여 안정성 확보 (패턴 70%, 전체 평균 30%)
        overall_avg = np.mean(self.binary_matrix, axis=0)
        return 0.7 * weighted_pred + 0.3 * overall_avg
    
    def get_scores(self) -> Dict[int, float]:
        probs = self.predict_probabilities()
        return {i+1: float(probs[i]) for i in range(45)}
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        probs = self.predict_probabilities()
        return sorted([int(idx + 1) for idx in np.argsort(probs)[-n_numbers:]])


def create_lstm_engine(numbers_matrix: np.ndarray, use_tensorflow: bool = False):
    """LSTM 엔진 생성 (리팩토링 버전은 기본적으로 경량 버전 사용)"""
    return LSTMEngine(numbers_matrix)
