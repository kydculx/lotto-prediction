"""
딥러닝 LSTM 엔진
- train() 호출 시: TensorFlow/Keras LSTM 모델 학습
- train() 미호출 시: numpy 기반 attention fallback (ml.py 메타 피처용 경량 모드)
"""
import os
import numpy as np
from typing import Dict, List, Optional
from .base import BaseEngine

import warnings
warnings.filterwarnings('ignore')

_TF = None
_LAYERS = None
_MODEL = None
_KCALLBACKS = None
TENSORFLOW_AVAILABLE: Optional[bool] = None


def _ensure_tensorflow() -> bool:
    """TensorFlow는 train() 호출 시점에만 lazy import. 모듈 import 시점 로드 시
    multiprocessing fork/spawn 후 자식 프로세스에서 tensorflow 재초기화가 dead-lock
    또는 수십 초 지연을 유발함 (특히 macOS)."""
    global _TF, _LAYERS, _MODEL, _KCALLBACKS, TENSORFLOW_AVAILABLE
    if TENSORFLOW_AVAILABLE is None:
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model, callbacks as keras_callbacks
            _TF = tf
            _LAYERS = layers
            _MODEL = Model
            _KCALLBACKS = keras_callbacks
            TENSORFLOW_AVAILABLE = True
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        except ImportError:
            TENSORFLOW_AVAILABLE = False
    return TENSORFLOW_AVAILABLE


class LSTMEngine(BaseEngine):
    """LSTM 예측 엔진 — train() 호출 시 TF Keras, 미호출 시 numpy fallback."""

    _model_cache: Dict[int, "LSTMEngine"] = {}
    _meta_cache: Dict[int, np.ndarray] = {}
    _meta_cache_loaded = False

    def __init__(self, numbers_matrix: np.ndarray, sequence_length: int = 10):
        super().__init__(numbers_matrix)
        self.sequence_length = sequence_length
        self.binary_matrix = self._create_binary_matrix()
        self.model = None
        self.is_trained = False
        self._cache_key = (self.n_draws // 50) * 50

    def _create_binary_matrix(self) -> np.ndarray:
        binary = np.zeros((self.n_draws, 45), dtype=np.float32)
        for i, row in enumerate(self.numbers_matrix):
            for num in row:
                binary[i, num - 1] = 1.0
        return binary

    def _numpy_fallback_scores(self) -> Dict[int, float]:
        L = min(self.sequence_length, self.n_draws - 1)
        if L < 2:
            avg = np.mean(self.binary_matrix, axis=0)
            return {i + 1: float(avg[i]) for i in range(45)}

        current_seq = self.binary_matrix[-L:].flatten()
        norm_curr = np.linalg.norm(current_seq)
        if norm_curr == 0:
            avg = np.mean(self.binary_matrix, axis=0)
            return {i + 1: float(avg[i]) for i in range(45)}

        similarities = []
        next_draws = []
        for i in range(self.n_draws - L):
            past_seq = self.binary_matrix[i:i + L].flatten()
            norm_past = np.linalg.norm(past_seq)
            if norm_past > 0:
                sim = np.dot(current_seq, past_seq) / (norm_curr * norm_past)
            else:
                sim = 0.0
            similarities.append(sim)
            next_draws.append(self.binary_matrix[i + L])

        if not similarities:
            avg = np.mean(self.binary_matrix, axis=0)
            return {i + 1: float(avg[i]) for i in range(45)}

        similarities = np.array(similarities)
        next_draws = np.array(next_draws)
        K = min(15, len(similarities))
        top_k = np.argsort(similarities)[-K:]
        weight_sum = np.sum(similarities[top_k])
        if weight_sum > 0:
            weighted = np.sum(next_draws[top_k] * similarities[top_k, np.newaxis], axis=0) / weight_sum
        else:
            weighted = np.mean(self.binary_matrix, axis=0)

        overall_avg = np.mean(self.binary_matrix, axis=0)
        final = 0.7 * weighted + 0.3 * overall_avg
        return {i + 1: float(final[i]) for i in range(45)}

    def _build_model(self):
        inputs = _LAYERS.Input(shape=(self.sequence_length, 45))
        x = _LAYERS.LSTM(64, return_sequences=True)(inputs)
        x = _LAYERS.LSTM(32)(x)
        x = _LAYERS.Dropout(0.2)(x)
        outputs = _LAYERS.Dense(45, activation='sigmoid')(x)
        model = _MODEL(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model

    def _prepare_data(self):
        X, y = [], []
        for i in range(self.sequence_length, self.n_draws):
            X.append(self.binary_matrix[i - self.sequence_length:i])
            y.append(self.binary_matrix[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _resolve_device(self) -> str:
        gpus = _TF.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    _TF.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            return '/GPU:0'
        return '/CPU:0'

    def train(self, epochs: int = 20, batch_size: int = 32, force: bool = False) -> bool:
        if not _ensure_tensorflow():
            return False

        if not force and self._cache_key in self._model_cache:
            cached = self._model_cache[self._cache_key]
            self.model = cached.model
            self.is_trained = True
            return True

        X, y = self._prepare_data()
        if len(X) < 100:
            return False

        device = self._resolve_device()
        try:
            _TF.keras.utils.set_random_seed(42)
            self.model = self._build_model()
            with _TF.device(device):
                self.model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[_KCALLBACKS.EarlyStopping(patience=3, restore_best_weights=True)]
                )
            self.is_trained = True
            self._model_cache[self._cache_key] = self
            self._evict_old_cache()
            return True
        except Exception:
            self.model = None
            self.is_trained = False
            return False

    @classmethod
    def _evict_old_cache(cls, keep_window: int = 100):
        if len(cls._model_cache) <= 4:
            return
        keys = sorted(cls._model_cache.keys())
        for k in keys[:-4]:
            if k < keys[-1] - keep_window:
                del cls._model_cache[k]

    def get_scores(self) -> Dict[int, float]:
        if not self.is_trained or self.model is None:
            return self._numpy_fallback_scores()

        try:
            last_seq = self.binary_matrix[-self.sequence_length:].reshape(
                1, self.sequence_length, 45
            )
            probs = self.model.predict(last_seq, verbose=0)[0]
            return {i + 1: float(probs[i]) for i in range(45)}
        except Exception:
            return self._numpy_fallback_scores()

    def predict(self, n_numbers: int = 6) -> List[int]:
        scores = self.get_scores()
        return sorted([num for num, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_numbers]])
