import json
import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.data_loader import LottoDataLoader
    from src.ensemble_predictor import EnsemblePredictor
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

def export_results():
    """분석 엔진을 실행하여 결과를 JSON 파일로 저장합니다."""
    print("🚀 분석 결과 내보내기 시작...")
    
    # 1. 데이터 로드
    loader = LottoDataLoader()
    # 데이터가 없으면 크롤링 수행 (내부 로직)
    loader.check_for_updates()
    
    matrix = loader.get_numbers_matrix()
    if matrix is None or len(matrix) == 0:
        print("❌ 분석할 데이터가 없습니다.")
        return

    # 2. 분석 실행 (Ensemble)
    print("🧠 엔진 분석 중 (100세트 생성)...")
    predictor = EnsemblePredictor(matrix)
    report = predictor.get_detailed_report(n_sets=100)
    
    # 3. 데이터 구조화 (Serializing)
    latest_round = int(loader.get_latest_round())
    prediction_data = {
        'latest_round': latest_round,
        'next_round': latest_round + 1,
        'hot_cold': report['hot_cold'],
        'engine_predictions': {k: [int(n) for n in v] for k, v in report['engine_predictions'].items()},
        'predicted_sets': [
            {'numbers': [int(n) for n in s[0]], 'confidence': float(s[1])}
            for s in report['predicted_sets']
        ],
        'sum_range': report['sum_range'],
        'export_time': Path(loader.file_path).stat().st_mtime if loader.file_path.exists() else 0
    }
    
    # 통계 데이터 추가 내보내기
    stats_data = {
        'total_draws': len(loader.df),
        'latest_draw': [int(n) for n in matrix[-1]],
        'rounds': loader.df['round'].tolist()[-50:], # 최근 50회차 리스트
    }

    # 4. JSON 파일 저장
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    pred_path = data_dir / "prediction.json"
    stats_path = data_dir / "stats.json"
    
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, ensure_ascii=False, indent=2)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 결과 저장 완료:")
    print(f"   - {pred_path}")
    print(f"   - {stats_path}")

if __name__ == "__main__":
    export_results()
