import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class LottoCrawler:
    """로또 당첨번호 공식 웹 크롤러 (동행복권) - 벌크 최적화 버전"""
    
    # 공식 사이트 AJAX API URL
    API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"
    REFERER_URL = "https://www.dhlottery.co.kr/lt645/result"
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            project_root = Path(__file__).parent.parent
            data_path = project_root / "data" / "lotto_results.json"
        self.data_path = Path(data_path)
        self.results = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': self.REFERER_URL,
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json, text/javascript, */*; q=0.01'
        }
        
    def load_existing_data(self):
        """기존 JSON 데이터를 로드합니다."""
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                self.results.sort(key=lambda x: x['round'])
                logger.info(f"✅ 기존 데이터 로드 완료: {len(self.results)}개 회차")
            except Exception as e:
                logger.error(f"⚠️ 데이터 로드 중 오류 발생: {e}")
                self.results = []

    def _parse_item(self, item: Dict) -> Dict:
        """API 응답 아이템을 공통 형식으로 파싱합니다."""
        raw_date = item['ltRflYmd']
        formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        return {
            "round": int(item['ltEpsd']),
            "date": formatted_date,
            "numbers": sorted([
                int(item['tm1WnNo']),
                int(item['tm2WnNo']),
                int(item['tm3WnNo']),
                int(item['tm4WnNo']),
                int(item['tm5WnNo']),
                int(item['tm6WnNo'])
            ]),
            "bonus": int(item['bnsWnNo'])
        }

    def fetch_all(self, force=False):
        """모든 회차 또는 누락된 회차를 벌크 API를 통해 한 번에 수집합니다."""
        self.load_existing_data()
        
        logger.info("📡 공식 사이트에서 전체 데이터를 조회 중입니다 (Bulk Fetch)...")
        try:
            params = {'srchLtEpsd': 'all'}
            response = requests.get(self.API_URL, params=params, headers=self.headers, timeout=20)
            if response.status_code != 200:
                logger.error(f"❌ API 연결 실패 (Status: {response.status_code})")
                return

            all_data = response.json()
            if not all_data.get('data') or not all_data['data'].get('list'):
                logger.error("❌ 유효한 데이터를 찾을 수 없습니다.")
                return

            raw_list = all_data['data']['list']
            # 전체 데이터를 파싱
            web_results = [self._parse_item(item) for item in raw_list]
            web_results.sort(key=lambda x: x['round'])
            
            latest_on_web = web_results[-1]['round'] if web_results else 0
            latest_stored = self.results[-1]['round'] if self.results else 0

            if not force and latest_stored >= latest_on_web:
                logger.info(f"✨ 이미 최신 상태입니다. (로컬: {latest_stored}, 웹: {latest_on_web})")
                return

            # 기존 데이터와 병합 (중복 제거 및 최신화)
            stored_rounds = {r['round'] for r in self.results}
            new_count = 0
            for item in web_results:
                if item['round'] not in stored_rounds:
                    self.results.append(item)
                    new_count += 1
            
            if new_count > 0:
                self.results.sort(key=lambda x: x['round'])
                self.save_data()
                logger.info(f"🎉 총 {new_count}개 회차의 누락된 데이터가 벌크로 업데이트되었습니다.")
            else:
                logger.info("💤 추가할 데이터가 없습니다.")

        except Exception as e:
            logger.error(f"❌ 전체 데이터 수집 중 오류 발생: {e}")

    def save_data(self):
        """수집된 데이터를 JSON 파일로 저장합니다."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 데이터 저장 완료: {self.data_path}")

if __name__ == "__main__":
    crawler = LottoCrawler()
    crawler.fetch_all()
