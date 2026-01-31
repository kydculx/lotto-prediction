import sqlite3
import json
from pathlib import Path
from datetime import datetime

class LottoDatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            project_root = Path(__file__).parent.parent
            db_path = project_root / "data" / "lotto.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """데이터베이스 및 테이블 초기화"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    round INTEGER PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_prediction(self, round_num, prediction_data):
        """회차별 분석 데이터를 저장합니다."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO predictions (round, data) VALUES (?, ?)",
                (round_num, json.dumps(prediction_data, ensure_ascii=False))
            )
            conn.commit()

    def get_prediction(self, round_num):
        """특정 회차의 분석 데이터를 가져옵니다."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT data FROM predictions WHERE round = ?", (round_num,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None

    def get_all_rounds(self):
        """데이터가 존재하는 모든 회차 리스트를 반환합니다."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT round FROM predictions ORDER BY round ASC")
            return [row[0] for row in cur.fetchall()]

    def save_meta(self, key, value):
        """메타데이터(통계 등)를 저장합니다."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False))
            )
            conn.commit()

    def get_meta(self, key):
        """메타데이터를 가져옵니다."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None
