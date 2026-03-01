
import sys
from pathlib import Path
import re

def apply_optimized_weights():
    # 학습 결과로 도출된 최적화 가중치 (과적합 방지 및 최신 트렌드 반영)
    best_weights = {
        'timeseries': 0.2000,          # 시계열 (추세) 유지
        'statistical': 0.1600,         # 통계 (빈도) 상향
        'sequence_correlation': 0.1500, # 수열 상관관계 상향
        'lstm': 0.1200,                # LSTM (딥러닝) 하향 조정 (과적합 방지)
        'numerology': 0.1100,          # 수비학 유지
        'advanced_pattern': 0.1000,    # 고급 패턴 상향
        'graph': 0.0800,               # 그래프 이론
        'gap': 0.0500,                 # 간격 분석
        'pattern': 0.0300              # 기본 패턴
    }
    
    predictor_path = Path(__file__).parent / "src" / "ensemble_predictor.py"
    
    with open(predictor_path, 'r') as f:
        content = f.read()
    
    # 가중치 문자열 생성
    weights_str = "    # 최적화된 엔진 가중치 (자동 최적화)\n    DEFAULT_WEIGHTS = {\n"
    for name, weight in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
        weights_str += f"        '{name}': {weight:.4f},\n"
    weights_str += "    }"
    
    # 정규식으로 교체
    pattern = r"    # 최적화된 엔진 가중치.*?DEFAULT_WEIGHTS = \{[^}]+\}"
    new_content = re.sub(pattern, weights_str, content, flags=re.DOTALL)
    
    with open(predictor_path, 'w') as f:
        f.write(new_content)
        
    print("✅ 최적화된 가중치가 적용되었습니다.")
    print("📊 변경 내역:")
    print(f"   - LSTM (Deep Learning): 0.16 -> 0.12 (과적합 방지)")
    print(f"   - Statistical (통계): 0.12 -> 0.16 (최신 빈도 반영)")
    print(f"   - Sequence (수열): 0.12 -> 0.15 (상관관계 강화)")

if __name__ == "__main__":
    apply_optimized_weights()
