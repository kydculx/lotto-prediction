#!/usr/bin/env python3
"""
학습된 가중치를 ensemble_predictor.py에 자동 적용
"""

import json
import re
from pathlib import Path


def apply_trained_weights():
    # 학습된 가중치 로드
    weights_file = Path(__file__).parent / "trained_weights_1000.json"
    
    if not weights_file.exists():
        print("❌ trained_weights_1000.json 파일이 없습니다.")
        print("   먼저 train_1000.py를 실행하세요.")
        return
    
    with open(weights_file, 'r') as f:
        data = json.load(f)
    
    weights = data['weights']
    score = data['best_score']
    
    print("\n" + "="*60)
    print("🔄 학습된 가중치 적용 중...")
    print("="*60)
    print(f"\n📊 학습 성능: {score:.4f}")
    print(f"📁 학습 범위: {data['training_rounds']}")
    print("\n최적화된 가중치:")
    
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 50)
        print(f"  {name:22s}: {weight:.4f} {bar}")
    
    # ensemble_predictor.py 경로
    predictor_path = Path(__file__).parent / "src" / "ensemble_predictor.py"
    
    with open(predictor_path, 'r') as f:
        content = f.read()
    
    # 가중치 문자열 생성
    weights_str = "    # 최적화된 엔진 가중치 (1~1000회차 학습 결과)\n"
    weights_str += "    DEFAULT_WEIGHTS = {\n"
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        weights_str += f"        '{name}': {weight:.4f},\n"
    weights_str += "    }"
    
    # 정규식으로 교체
    pattern = r"    # 최적화된 엔진 가중치.*?DEFAULT_WEIGHTS = \{[^}]+\}"
    new_content = re.sub(pattern, weights_str, content, flags=re.DOTALL)
    
    # 백업 생성
    backup_path = predictor_path.with_suffix('.py.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # 새 가중치 적용
    with open(predictor_path, 'w') as f:
        f.write(new_content)
    
    print("\n" + "="*60)
    print("✅ 가중치 적용 완료!")
    print("="*60)
    print(f"\n📝 백업 파일: {backup_path}")
    print(f"📝 적용 파일: {predictor_path}")
    print("\n💡 이제 main.py를 실행하면 새로운 가중치로 예측합니다!")
    print("   예: python3 main.py --predict")


if __name__ == "__main__":
    apply_trained_weights()
