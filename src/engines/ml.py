"""
XGBoost 기계학습 예측 엔진 (간소화 버전)
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from .base import BaseEngine
import warnings
warnings.filterwarnings('ignore')


class MLEngine(BaseEngine):
    """ML 기반 예측 엔진 (XGBoost 대체용 GradientBoosting)"""
    
    def __init__(self, numbers_matrix: np.ndarray, lookback: int = 10):
        super().__init__(numbers_matrix)
        self.lookback = lookback
        self.model = None
        
    def _extract_features(self, idx: int) -> np.ndarray:
        if idx < self.lookback: return None
        features, recent = [], self.numbers_matrix[idx - self.lookback:idx]
        freq = Counter(recent.flatten())
        for num in range(1, 46): features.append(freq.get(num, 0))
        for num in range(1, 46):
            gap = self.lookback
            for i, row in enumerate(reversed(recent)):
                if num in row: gap = i; break
            features.append(gap)
        binary_last = np.zeros(45)
        for num in self.numbers_matrix[idx - 1]: binary_last[num - 1] = 1
        features.extend(binary_last)
        features.append(np.mean([sum(row) for row in recent]))
        features.append(np.mean([sum(1 for n in row if n % 2 == 1) for row in recent]))
        return np.array(features)
    
    def train(self, n_estimators: int = 50) -> bool:
        try:
            from sklearn.multioutput import MultiOutputClassifier
            from sklearn.ensemble import GradientBoostingClassifier
            X, y = [], []
            binary = np.zeros((self.n_draws, 45), dtype=np.float32)
            for i, row in enumerate(self.numbers_matrix):
                for num in row: binary[i, num - 1] = 1.0
            for i in range(self.lookback, self.n_draws):
                f = self._extract_features(i)
                if f is not None: X.append(f); y.append(binary[i])
            if len(X) < 100: return False
            self.model = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=n_estimators, max_depth=3, random_state=42), n_jobs=-1)
            self.model.fit(X, y)
            return True
        except ImportError: return False
    
    def get_scores(self) -> Dict[int, float]:
        if self.model is None and not self.train(20):
            recent = self.numbers_matrix[-50:].flatten()
            freq = Counter(recent)
            return {i: freq.get(i, 0)/50 for i in range(1, 46)}
        
        f = self._extract_features(self.n_draws).reshape(1, -1)
        proba = self.model.predict_proba(f)
        scores = {i+1: float(p[0][1] if len(p[0]) > 1 else 0.5) for i, p in enumerate(proba)}
        max_s = max(scores.values()) or 1
        return {k: v / max_s for k, v in scores.items()}
    
    def predict(self, n_numbers: int = 6) -> List[int]:
        scores = self.get_scores()
        return sorted([num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_numbers]])
