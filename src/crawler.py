import requests
from bs4 import BeautifulSoup
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

class LottoCrawler:
    """로또 당첨번호 웹 크롤러"""
    
    BASE_URL = "https://www.lotto.co.kr/article/list/AC01"
    AJAX_URL = "https://www.lotto.co.kr/lotto_info/list_ajax"
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            project_root = Path(__file__).parent.parent
            data_path = project_root / "data" / "lotto_results.json"
        self.data_path = Path(data_path)
        self.results = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Referer': self.BASE_URL
        }
    
    def load_existing_data(self):
        """기존 JSON 데이터를 로드합니다."""
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                # 회차 순으로 정렬
                self.results.sort(key=lambda x: x['round'])
                print(f"✅ 기존 데이터 로드 완료: {len(self.results)}개 회차")
            except Exception as e:
                print(f"⚠️ 데이터 로드 중 오류 발생: {e}")
                self.results = []
    
    def get_latest_round_num(self) -> int:
        """웹사이트에서 가장 최신 회차 번호를 가져옵니다."""
        try:
            # 첫 페이지 AJAX 요청으로 최신 회차 확인
            payload = "category=AC01&startPos=0&endPos=10&pageSize=10&page=1"
            response = requests.post(self.AJAX_URL, data=payload, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                first_round_span = soup.select_one('.wnr_cur_list li span')
                if first_round_span:
                    import re
                    match = re.search(r'(\d+)회', first_round_span.get_text())
                    if match:
                        return int(match.group(1))
        except Exception as e:
            print(f"❌ 최신 회차 확인 실패: {e}")
        return 0

    def parse_html_fragment(self, html: str) -> List[Dict]:
        """AJAX 응답 HTML에서 데이터를 추출합니다."""
        soup = BeautifulSoup(html, 'html.parser')
        new_data = []
        
        # <ul> 내의 <li> 항목들 파싱
        items = soup.select('.wnr_cur_list li')
        
        for item in items:
            try:
                spans = item.select('span')
                if not spans: continue
                
                # 회차 번호 (첫 번째 span)
                round_text = spans[0].get_text()
                import re
                round_match = re.search(r'(\d+)회', round_text)
                if not round_match: continue
                round_num = int(round_match.group(1))
                
                # 날짜 (두 번째 span)
                draw_date = spans[1].get_text() if len(spans) > 1 else ""
                
                # 당첨 번호 추출 (img 태그의 src 또는 alt)
                # 실제 번호는 src의 파일명에 있음 (예: /.../6.png -> 6)
                imgs = item.select('.cur_wnr_item img')
                numbers = []
                bonus = None
                
                # 이미지 src 파일명에서 숫자 추출
                for img in imgs:
                    src = img.get('src', '')
                    import os
                    filename = os.path.basename(src)
                    # 파일명에서 숫자 추출 (예: 6.png -> 6)
                    num_match = re.search(r'(\d+)', filename)
                    
                    if num_match:
                        num = int(num_match.group(1))
                        # 이미지는 보통 7개가 나옴 (6개 당첨 + 1개 보너스)
                        # 'servic' 등으로 끝나는 이미지는 제외 처리 필요 (필터링)
                        if 'lottoball' in src:
                            if len(numbers) < 6:
                                numbers.append(num)
                            else:
                                bonus = num
                
                if len(numbers) == 6:
                    new_data.append({
                        "round": round_num,
                        "date": draw_date,
                        "numbers": sorted(numbers),
                        "bonus": bonus
                    })
            except Exception as e:
                print(f"⚠️ 파싱 중 항목 건너뜀 (회차 미상): {e}")
                
        return new_data

    def fetch_all(self, force=False):
        """전체 회차를 수집하거나 신규 회차만 수집합니다."""
        self.load_existing_data()
        latest_on_web = self.get_latest_round_num()
        latest_stored = self.results[-1]['round'] if self.results else 0
        
        if not force and latest_stored >= latest_on_web:
            print(f"✨ 최신 상태입니다. (로컬: {latest_stored}, 웹: {latest_on_web})")
            return
        
        print(f"🚀 데이터 수집 시작... (웹 최신: {latest_on_web}회)")
        
        all_new_results = []
        page = 1
        page_size = 10
        total_collected = 0
        
        while True:
            print(f"📥 페이지 {page} 수집 중...")
            start_pos = (page - 1) * page_size
            end_pos = page * page_size
            
            payload = f"category=AC01&startPos={start_pos}&endPos={end_pos}&pageSize={page_size}&total={latest_on_web}&page={page}"
            
            try:
                response = requests.post(self.AJAX_URL, data=payload, headers=self.headers, timeout=10)
                if response.status_code != 200 or not response.text.strip():
                    break
                
                page_data = self.parse_html_fragment(response.text)
                if not page_data:
                    break
                
                # 중복 확인 및 종료 조건
                stop_crawling = False
                for d in page_data:
                    if not any(r['round'] == d['round'] for r in self.results) and \
                       not any(r['round'] == d['round'] for r in all_new_results):
                        all_new_results.append(d)
                    else:
                        if not force:
                            stop_crawling = True
                            break
                
                if stop_crawling:
                    break
                    
                page += 1
                time.sleep(0.5) # 서버 부하 방지
            except Exception as e:
                print(f"❌ 페이지 {page} 수집 중 오류: {e}")
                break
        
        if all_new_results:
            self.results.extend(all_new_results)
            self.results.sort(key=lambda x: x['round'])
            self.save_data()
            print(f"🎉 수집 완료! {len(all_new_results)}개 신규 회차 추가됨.")
        else:
            print("💤 추가할 데이터가 없습니다.")

    def save_data(self):
        """수집된 데이터를 JSON으로 저장합니다."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"💾 데이터 저장 완료: {self.data_path}")

if __name__ == "__main__":
    crawler = LottoCrawler()
    # 처음 실행 시에는 force=True로 전체 수집 가능, 이후에는 False로 증분 업데이트
    crawler.fetch_all(force=False)
