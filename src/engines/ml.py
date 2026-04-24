"""
XGBoost 기계학습 예측 엔진 (간소화 버전)
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from .base import BaseEngine
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


class MLEngine(BaseEngine):
    """ML 기반 예측 엔진 (XGBoost 대체용 GradientBoosting)"""
    
    _model_cache = {} # 백테스팅 속도 최적화를 위한 전역 모델 캐시
    _meta_cache = {}
    _meta_cache_file = "data/ml_meta_features.pkl"
    _meta_cache_loaded = False
    
    @classmethod
    def _load_meta_cache(cls):
        if not cls._meta_cache_loaded:
            if os.path.exists(cls._meta_cache_file):
                try:
                    with open(cls._meta_cache_file, 'rb') as f:
                        cls._meta_cache = pickle.load(f)
                except Exception:
                    pass
            cls._meta_cache_loaded = True

    @classmethod
    def _save_meta_cache(cls):
        try:
            os.makedirs(os.path.dirname(cls._meta_cache_file), exist_ok=True)
            with open(cls._meta_cache_file, 'wb') as f:
                pickle.dump(cls._meta_cache, f)
        except Exception:
            pass

    @classmethod
    def _get_meta_features(cls, idx: int, matrix: np.ndarray) -> np.ndarray:
        if idx in cls._meta_cache:
            return cls._meta_cache[idx]
            
        subset = matrix[:idx]
        if len(subset) < 10:
            return np.zeros(45 * 5, dtype=np.float32)
            
        from .fourier import FourierEngine
        from .advanced_pattern import AdvancedPatternEngine
        from .statistical import StatisticalEngine
        from .lstm import LSTMEngine
        from .poisson import PoissonEngine
        
        engines = [
            FourierEngine(subset),
            AdvancedPatternEngine(subset),
            StatisticalEngine(subset),
            LSTMEngine(subset),
            PoissonEngine(subset)
        ]
        
        meta_f = []
        for eng in engines:
            try:
                scores = eng.get_scores()
                meta_f.extend([scores.get(num, 0.0) for num in range(1, 46)])
            except Exception:
                meta_f.extend([0.0] * 45)
                
        meta_arr = np.array(meta_f, dtype=np.float32)
        cls._meta_cache[idx] = meta_arr
        return meta_arr

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
        
        # Meta-Features 추출 (다른 5개 주요 엔진들의 예측 점수)
        meta_features = self.__class__._get_meta_features(idx, self.numbers_matrix)
        
        return np.concatenate([np.array(features, dtype=np.float32), meta_features])
    
    def train(self, n_estimators: int = 30) -> bool:
        # 백테스팅 시 매 회차 재학습하는 오버헤드 방지 (50회차 단위 모델 재사용)
        # Lookahead Bias(미래 참조 오류)가 없도록 현재 데이터 수 이하의 키만 사용
        cache_key = (self.n_draws // 50) * 50
        if cache_key in self.__class__._model_cache:
            self.model = self.__class__._model_cache[cache_key]
            return True
            
        try:
            from sklearn.multioutput import MultiOutputClassifier
            from sklearn.ensemble import RandomForestClassifier
            self.__class__._load_meta_cache()
            initial_cache_size = len(self.__class__._meta_cache)
            
            X, y = [], []
            binary = np.zeros((self.n_draws, 45), dtype=np.float32)
            for i, row in enumerate(self.numbers_matrix):
                for num in row: binary[i, num - 1] = 1.0
            for i in range(self.lookback, self.n_draws):
                f = self._extract_features(i)
                if f is not None: X.append(f); y.append(binary[i])
            if len(X) < 100: return False
            
            if len(self.__class__._meta_cache) > initial_cache_size:
                self.__class__._save_meta_cache()
            
            # GradientBoosting보다 훨씬 빠르고 병렬 처리가 잘 되는 RandomForest 사용
            # 메타 피처가 늘어났으므로 capacity 증대 (n_estimators=50, max_depth=8)
            self.model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=1), n_jobs=-1)
            self.model.fit(X, y)
            
            # 모델 캐싱 및 메모리 관리
            self.__class__._model_cache[cache_key] = self.model
            old_keys = [k for k in self.__class__._model_cache.keys() if k < cache_key - 100]
            for k in old_keys:
                del self.__class__._model_cache[k]
                
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
