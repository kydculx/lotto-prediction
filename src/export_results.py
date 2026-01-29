import json
import logging
import sys
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.data_loader import LottoDataLoader
    from src.ensemble_predictor import EnsemblePredictor
except ImportError as e:
    logger.error(f"모듈 임포트 실패: {e}")
    sys.exit(1)

def calculate_frequencies(loader):
    """번호별 출현 빈도를 계산하여 딕셔너리로 반환합니다."""
    all_numbers = loader.get_all_numbers_flat()
    unique, counts = np.unique(all_numbers, return_counts=True)
    
    freq_dict = {int(i): 0 for i in range(1, 46)}
    for num, count in zip(unique, counts):
        freq_dict[int(num)] = int(count)
    return freq_dict

def export_results():
    """분석 엔진을 실행하고 결과를 JSON 파일로 저장합니다."""
    logger.info("🚀 분석 결과 내보내기 시작...")
    
    # 1. 데이터 로드 및 업데이트 체크
    loader = LottoDataLoader()
    loader.check_for_updates()
    
    matrix = loader.get_numbers_matrix()
    if matrix is None or len(matrix) == 0:
        logger.error("분석할 데이터가 없습니다.")
        return

    # 2. AI 엔진 분석 실행
    logger.info("🧠 AI 엔진 분석 중 (100세트 생성)...")
    predictor = EnsemblePredictor(matrix)
    report = predictor.get_detailed_report(n_sets=100)
    
    # 3. 데이터 구조화
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
    
    stats_data = {
        'total_draws': len(loader.df),
        'latest_draw': [int(n) for n in matrix[-1]],
        'rounds': loader.df['round'].tolist()[-50:],
    }

    # 4. 파일 저장 설정
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_save = {
        "prediction.json": prediction_data,
        "stats.json": stats_data,
        "frequencies.json": calculate_frequencies(loader)
    }

    for filename, content in files_to_save.items():
        file_path = data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 저장 완료: {file_path}")

    logger.info("✅ 모든 분석 결과 내보내기가 완료되었습니다.")

if __name__ == "__main__":
    try:
        export_results()
    except Exception as e:
        logger.exception(f"내보내기 중 예기치 않은 오류 발생: {e}")
        sys.exit(1)
