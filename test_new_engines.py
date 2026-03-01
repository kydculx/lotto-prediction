
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LottoDataLoader
from src.engines.poisson import PoissonEngine
from src.engines.fourier import FourierEngine

def test_engines():
    loader = LottoDataLoader()
    loader.load()
    matrix = loader.get_numbers_matrix()
    
    print(f"Data draws: {len(matrix)}")
    
    print("\n--- Testing PoissonEngine ---")
    pe = PoissonEngine(matrix)
    p_scores = pe.get_scores()
    p_pred = pe.predict()
    print(f"Poisson Prediction: {p_pred}")
    print(f"Top 5 scores: {sorted(p_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    print("\n--- Testing FourierEngine ---")
    fe = FourierEngine(matrix)
    f_scores = fe.get_scores()
    f_pred = fe.predict()
    print(f"Fourier Prediction: {f_pred}")
    print(f"Top 5 scores: {sorted(f_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")

if __name__ == "__main__":
    test_engines()
