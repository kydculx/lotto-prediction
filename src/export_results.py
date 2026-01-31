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
    from src.database_manager import LottoDatabaseManager
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

def export_results(target_round=None, round_range=None):
    """분석 엔진을 실행하고 결과를 JSON 및 SQLite에 저장합니다."""
    
    # 1. 데이터 로드 및 DB 매니저 초기화
    loader = LottoDataLoader()
    loader.check_for_updates()
    db_manager = LottoDatabaseManager()
    
    all_rounds_df = loader.df.copy()
    max_round = int(all_rounds_df['round'].max())
    
    # 처리할 회차 리스트 결정
    targets = []
    if target_round:
        targets = [target_round]
    elif round_range:
        start, end = round_range
        targets = list(range(start, end + 1))
    else:
        # 기본값: 최신 회차 결과 + 누락된 역사적 데이터 자동 수집
        targets = [None]
        
        # 누락된 역사적 JSON 파일 확인 (회차 1부터 max_round-1까지)
        history_dir = PROJECT_ROOT / "data" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        missing_history = []
        for r in range(1, max_round):
            json_file = history_dir / f"prediction_{r+1}.json"
            if not json_file.exists():
                missing_history.append(r)
        
        if missing_history:
            logger.info(f"🔍 누락된 역사적 데이터 {len(missing_history)}개를 발견했습니다. 자동으로 내보내기를 수행합니다.")
            targets = missing_history + targets

    for current_target in targets:
        # 1-1. 분석 대상 회차 및 다음 회차 번호 계산
        if current_target:
            target_round_num = current_target
            analysis_round_num = target_round_num # target_round_num 데이터까지 보고 target_round_num+1을 예측
        else:
            target_round_num = max_round
            analysis_round_num = target_round_num

        # 1-2. DB 확인 (이미 분석된 데이터가 있으면 익스포트만 수행)
        existing_data = db_manager.get_prediction(target_round_num)
        
        if existing_data:
            logger.info(f"⏭️ {target_round_num}회차 데이터가 DB에 이미 존재합니다. 익스포트만 수행합니다.")
            prediction_data = existing_data
        else:
            # 신규 분석 수행
            if current_target:
                logger.info(f"📍 {target_round_num}회차 시점 분석 중...")
                loader.df = all_rounds_df[all_rounds_df['round'] <= target_round_num].copy()
                loader.numbers_df = loader.df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].copy()
            else:
                logger.info("🚀 최신 회차 분석 중...")
                loader.df = all_rounds_df.copy()
                loader.numbers_df = loader.df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].copy()

            matrix = loader.get_numbers_matrix()
            if matrix is None or len(matrix) == 0:
                logger.warning(f"{target_round_num}회차: 분석할 데이터가 부족하여 건너뜜")
                continue

            # 2. AI 엔진 분석 실행
            predictor = EnsemblePredictor(matrix)
            report = predictor.get_detailed_report(n_sets=100)
            
            # 3. 데이터 구조화
            prediction_data = {
                'latest_round': target_round_num,
                'next_round': target_round_num + 1,
                'hot_cold': report['hot_cold'],
                'engine_predictions': {k: [int(n) for n in v] for k, v in report['engine_predictions'].items()},
                'predicted_sets': [
                    {'numbers': [int(n) for n in s[0]], 'confidence': float(s[1])}
                    for s in report['predicted_sets']
                ],
                'sum_range': report['sum_range'],
                'export_time': Path(loader.file_path).stat().st_mtime if loader.file_path.exists() else 0
            }
            
            # DB 저장
            db_manager.save_prediction(target_round_num, prediction_data)
        
        # 4. JSON 파일 익스포트 (정적 사이트 호환용)
        is_historical = target_round_num < max_round
        
        if is_historical:
            data_dir = PROJECT_ROOT / "data" / "history"
            prediction_filename = f"prediction_{target_round_num + 1}.json"
        else:
            data_dir = PROJECT_ROOT / "data"
            prediction_filename = "prediction.json"
            
        data_dir.mkdir(parents=True, exist_ok=True)

        # JSON 저장
        file_path = data_dir / prediction_filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
        
        if not is_historical:
            # 최신 회차일 때만 stats.json과 frequencies.json 업데이트 및 DB 저장
            stats_data = {
                'total_draws': len(all_rounds_df),
                'latest_draw': [int(n) for n in all_rounds_df.iloc[-1][['num1','num2','num3','num4','num5','num6']].values],
                'rounds': all_rounds_df['round'].tolist()[-50:],
            }
            freq_data = calculate_frequencies(loader)
            
            # 파일 저장
            with open(data_dir / "stats.json", 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            with open(data_dir / "frequencies.json", 'w', encoding='utf-8') as f:
                json.dump(freq_data, f, ensure_ascii=False, indent=2)
                
            # DB 메타 저장
            db_manager.save_meta("stats", stats_data)
            db_manager.save_meta("frequencies", freq_data)
            
            logger.info(f"💾 최신 데이터 및 통계 저장 완료 (DB & JSON)")
        else:
            logger.debug(f"💾 역사적 데이터 익스포트 완료: {file_path}")

    logger.info("✅ 모든 요청된 데이터 처리가 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lotto AI 분석 결과 내보내기")
    parser.add_argument("--round", type=int, help="분석 시점으로 지정할 회차 (예: 100)")
    parser.add_argument("--range", type=str, help="분석할 회차 범위 (예: 1-100)")
    args = parser.parse_args()
    
    round_range = None
    if args.range:
        try:
            start, end = map(int, args.range.split('-'))
            round_range = (start, end)
        except ValueError:
            logger.error("범위 형식이 잘못되었습니다. (예: 1-100)")
            sys.exit(1)

    try:
        export_results(target_round=args.round, round_range=round_range)
    except Exception as e:
        logger.exception(f"내보내기 중 예기치 않은 오류 발생: {e}")
        sys.exit(1)
