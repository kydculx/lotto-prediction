"""
데이터 로더 모듈
로또 당첨번호 Excel 파일을 로드하고 전처리합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List


class LottoDataLoader:
    """로또 당첨번호 데이터 로더"""
    
    def __init__(self, file_path: str = None):
        """
        Args:
            file_path: 데이터 파일 경로 (기본값: 'data/lotto_results.json' 또는 '로또 회차별 당첨번호.xlsx')
        """
        project_root = Path(__file__).parent.parent
        self.json_path = project_root / "data" / "lotto_results.json"
        self.excel_path = project_root / "로또 회차별 당첨번호.xlsx"
        
        if file_path:
            self.file_path = Path(file_path)
        else:
            # JSON 우선, 없으면 엑셀 사용
            self.file_path = self.json_path if self.json_path.exists() else self.excel_path
            
        self.df = None
        self.numbers_df = None
        self.last_mtime = 0
        self.last_web_check = 0 # 마지막 웹 확인 시간
        self.sync_interval = 3600 # 웹 확인 주기 (1시간)
        
    def check_for_updates(self):
        """파일이 수정되었는지 확인하거나 최신 데이터를 웹에서 확인합니다."""
        import time
        now = time.time()
        
        # 1. 파일 부재 시 즉시 크롤링
        if not self.file_path.exists() and not self.json_path.exists():
            print("⚠️ 데이터 파일이 없습니다. 크롤링을 시도합니다...")
            self.run_crawler()
            self.last_web_check = now
            return

        # 2. 일정 주기가 지났으면 웹사이트 최신 회차 확인 (자동 동기화)
        if now - self.last_web_check > self.sync_interval:
            print("🌐 웹사이트 동기화 확인 중...")
            self.run_crawler()
            self.last_web_check = now

        # 3. 로컬 파일 수정 여부 확인 (JSON/Excel)
        if self.file_path.exists():
            current_mtime = self.file_path.stat().st_mtime
            if current_mtime > self.last_mtime:
                print(f"🔄 데이터 변경 감지: {self.file_path.name} 로드 중...")
                self.load()
                self.last_mtime = current_mtime

    def run_crawler(self):
        """웹 크롤러를 실행하여 데이터를 최신화합니다."""
        try:
            from src.crawler import LottoCrawler
            crawler = LottoCrawler()
            crawler.fetch_all(force=False)
            # 수집 후 파일 경로를 JSON으로 전환
            if self.json_path.exists():
                self.file_path = self.json_path
                self.load()
        except ImportError:
            print("❌ Crawler 모듈을 찾을 수 없습니다.")
        except Exception as e:
            print(f"❌ 크롤링 중 오류 발생: {e}")

    def load(self) -> pd.DataFrame:
        """JSON 또는 Excel 파일을 로드하고 전처리합니다."""
        if self.file_path.suffix == '.json':
            import json
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 데이터를 DataFrame으로 변환
            # JSON 포맷: [{"round": 1, "date": "...", "numbers": [1,2,3...], "bonus": 7}, ...]
            rows = []
            for d in data:
                row = {
                    'round': d['round'],
                    'num1': d['numbers'][0],
                    'num2': d['numbers'][1],
                    'num3': d['numbers'][2],
                    'num4': d['numbers'][3],
                    'num5': d['numbers'][4],
                    'num6': d['numbers'][5],
                    'bonus': d['bonus']
                }
                rows.append(row)
            self.df = pd.DataFrame(rows)
        else:
            # 원본 Excel 데이터 로드
            self.df = pd.read_excel(self.file_path)
            # 컬럼명 정규화 (필요시)
            if '당첨번호' in self.df.columns:
                self.df = pd.DataFrame({
                    'round': self.df['회차'],
                    'num1': self.df['당첨번호'],
                    'num2': self.df['Unnamed: 3'],
                    'num3': self.df['Unnamed: 4'],
                    'num4': self.df['Unnamed: 5'],
                    'num5': self.df['Unnamed: 6'],
                    'num6': self.df['Unnamed: 7'],
                    'bonus': self.df['보너스번호'],
                })
        
        self.last_mtime = self.file_path.stat().st_mtime if self.file_path.exists() else 0
        
        # 회차 기준 정렬 (오름차순)
        self.df = self.df.sort_values('round').reset_index(drop=True)
        # 숫자만 추출한 배열 (분석용)
        self.numbers_df = self.df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].copy()
        
        return self.df
    
    def get_all_numbers_flat(self) -> np.ndarray:
        """모든 당첨번호를 1차원 배열로 반환 (보너스 제외)"""
        self.check_for_updates()
        if self.numbers_df is None:
            self.load()
        return self.numbers_df.values.flatten()
    
    def get_numbers_matrix(self) -> np.ndarray:
        """당첨번호를 2D 배열로 반환 (회차 x 6개 번호)"""
        self.check_for_updates()
        if self.numbers_df is None:
            self.load()
        return self.numbers_df.values
    
    def get_recent_draws(self, n: int = 50) -> pd.DataFrame:
        """최근 n회차 데이터 반환"""
        self.check_for_updates()
        if self.df is None:
            self.load()
        return self.df.tail(n).copy()
    
    def get_binary_matrix(self) -> np.ndarray:
        """
        멀티-핫 인코딩 매트릭스 반환
        Shape: (회차수, 45) - 각 번호 출현 여부
        """
        self.check_for_updates()
        if self.numbers_df is None:
            self.load()
            
        n_draws = len(self.numbers_df)
        binary_matrix = np.zeros((n_draws, 45), dtype=np.int8)
        
        for i, row in enumerate(self.numbers_df.values):
            for num in row:
                binary_matrix[i, num - 1] = 1
                
        return binary_matrix
    
    def get_latest_round(self) -> int:
        """가장 최근 회차 번호 반환"""
        self.check_for_updates()
        if self.df is None:
            self.load()
        return int(self.df['round'].max())
    
    def get_draw_by_round(self, round_num: int) -> List[int]:
        """특정 회차의 당첨번호 반환"""
        self.check_for_updates()
        if self.df is None:
            self.load()
        row = self.df[self.df['round'] == round_num]
        if len(row) == 0:
            return None
        return row[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values[0].tolist()


# 테스트 코드
if __name__ == "__main__":
    loader = LottoDataLoader()
    df = loader.load()
    print(f"총 {len(df)}회차 데이터 로드 완료")
    print(f"최근 회차: {loader.get_latest_round()}")
    print(f"\n최근 5회차 데이터:")
    print(loader.get_recent_draws(5))
