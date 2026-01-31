import json
import logging
import sys
import argparse
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

def export_results(target_round=None):
    """분석 엔진을 실행하고 결과를 JSON 파일로 저장합니다."""
    logger.info("🚀 분석 결과 내보내기 시작...")
    
    # 1. 데이터 로드 및 업데이트 체크
    loader = LottoDataLoader()
    loader.check_for_updates()
    
    # 2. 회차 지정 시 해당 회차까지만 데이터 슬라이싱
    if target_round:
        logger.info(f"📍 {target_round}회차 시점 분석 모드 활성화")
        loader.load() # 명시적 로드
        if target_round not in loader.df['round'].values:
            logger.error(f"회차 {target_round}을 찾을 수 없습니다.")
            return
        # 해당 회차까지의 데이터만 남김
        loader.df = loader.df[loader.df['round'] <= target_round]
        loader.numbers_df = loader.df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].copy()
    
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
    if target_round:
        data_dir = data_dir / "history"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    prediction_filename = f"prediction_{latest_round + 1}.json" if target_round else "prediction.json"
    
    files_to_save = {
        prediction_filename: prediction_data,
        "stats.json": stats_data,
        "frequencies.json": calculate_frequencies(loader)
    }

    # 역사적 데이터 생성 시 stats와 frequencies는 최신 파일을 덮어쓰지 않도록 (검색용 prediction만 저장)
    save_list = [prediction_filename] if target_round else files_to_save.keys()

    for filename in save_list:
        file_path = data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(files_to_save[filename], f, ensure_ascii=False, indent=2)
        logger.info(f"💾 저장 완료: {file_path}")

    logger.info("✅ 모든 분석 결과 내보내기가 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lotto AI 분석 결과 내보내기")
    parser.add_argument("--round", type=int, help="분석 시점으로 지정할 회차 (예: 100)")
    args = parser.parse_args()
    
    try:
        export_results(target_round=args.round)
    except Exception as e:
        logger.exception(f"내보내기 중 예기치 않은 오류 발생: {e}")
        sys.exit(1)
