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
    """로또 당첨번호 공식 웹 크롤러 (동행복권)"""
    
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

    def get_latest_round_num(self) -> int:
        """공식 API에서 가장 최신 회차 번호를 가져옵니다."""
        try:
            # srchLtEpsd=all 을 사용하여 최근 결과들을 가져옴
            params = {'srchLtEpsd': 'all'}
            response = requests.get(self.API_URL, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and data['data'].get('list'):
                    # 리스트에서 가장 큰 ltEpsd 값을 찾음
                    rounds = [int(item['ltEpsd']) for item in data['data']['list']]
                    return max(rounds) if rounds else 0
        except Exception as e:
            logger.error(f"❌ 최신 회차 번호 가져오기 실패: {e}")
        return 0

    def fetch_round(self, round_num: int) -> Optional[Dict]:
        """특정 회차의 데이터를 공식 API에서 가져옵니다."""
        try:
            params = {'srchLtEpsd': str(round_num)}
            response = requests.get(self.API_URL, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                res_data = response.json()
                if res_data.get('data') and res_data['data'].get('list'):
                    item = res_data['data']['list'][0]
                    # 날짜 형식 변환: 20260124 -> 2026-01-24
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
        except Exception as e:
            logger.error(f"❌ {round_num}회차 데이터 가져오기 실패: {e}")
        return None

    def fetch_all(self, force=False):
        """기존 데이터에 없는 최신 회차들을 수집합니다."""
        self.load_existing_data()
        latest_on_web = self.get_latest_round_num()
        
        if latest_on_web == 0:
            logger.error("웹에서 최신 회차 정보를 읽어올 수 없습니다.")
            return

        latest_stored = self.results[-1]['round'] if self.results else 0
        
        if not force and latest_stored >= latest_on_web:
            logger.info(f"✨ 이미 최신 상태입니다. (로컬: {latest_stored}, 웹: {latest_on_web})")
            return
        
        logger.info(f"🚀 {latest_stored + 1}회부터 {latest_on_web}회까지 수집을 시작합니다.")
        
        new_results = []
        for r_num in range(latest_stored + 1, latest_on_web + 1):
            logger.info(f"📥 {r_num}회차 응답 대기 중...")
            data = self.fetch_round(r_num)
            if data:
                new_results.append(data)
                # API 부하 방지를 위해 아주 약간의 지연
                time.sleep(0.2)
            else:
                logger.warning(f"⚠️ {r_num}회차 수집 실패")
        
        if new_results:
            self.results.extend(new_results)
            self.results.sort(key=lambda x: x['round'])
            self.save_data()
            logger.info(f"🎉 총 {len(new_results)}개 회차 업데이트 완료!")
        else:
            logger.info("💤 업데이트할 데이터가 없습니다.")

    def save_data(self):
        """수집된 데이터를 JSON 파일로 저장합니다."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 데이터 저장 완료: {self.data_path}")

if __name__ == "__main__":
    crawler = LottoCrawler()
    crawler.fetch_all()
