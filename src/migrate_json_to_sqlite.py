import json
from pathlib import Path
from src.database_manager import LottoDatabaseManager
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def migrate():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    history_dir = data_dir / "history"
    db_manager = LottoDatabaseManager()

    # 1. 역사적 데이터 마이그레이션 (data/history/*.json)
    if history_dir.exists():
        logger.info("📂 역사적 데이터 마이그레이션 시작...")
        for json_file in history_dir.glob("prediction_*.json"):
            try:
                round_num = int(json_file.stem.split('_')[1]) - 1 # 파일명 prediction_101.json은 100회차 결과
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                db_manager.save_prediction(round_num, data)
                logger.info(f"✅ {round_num}회차 저장 완료")
            except Exception as e:
                logger.error(f"❌ {json_file.name} 처리 중 오류: {e}")

    # 2. 최신 데이터 마이그레이션 (data/prediction.json)
    prediction_json = data_dir / "prediction.json"
    if prediction_json.exists():
        try:
            with open(prediction_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            round_num = data['latest_round']
            db_manager.save_prediction(round_num, data)
            logger.info(f"✅ 최신({round_num}회차) 데이터 저장 완료")
        except Exception as e:
            logger.error(f"❌ prediction.json 처리 중 오류: {e}")

    # 3. 메타 데이터 마이그레이션 (stats.json, frequencies.json)
    for meta_key in ["stats", "frequencies"]:
        meta_file = data_dir / f"{meta_key}.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                db_manager.save_meta(meta_key, data)
                logger.info(f"✅ 메타({meta_key}) 데이터 저장 완료")
            except Exception as e:
                logger.error(f"❌ {meta_file.name} 처리 중 오류: {e}")

    logger.info("🚀 마이그레이션 완료!")

if __name__ == "__main__":
    migrate()
