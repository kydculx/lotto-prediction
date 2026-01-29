#!/bin/bash
# Lotto AI Server Runner (Double-clickable for Mac)

# 현재 스크립트 위치로 이동
cd "$(dirname "$0")"

echo "================================================"
echo "🎯 로또 AI 프리미엄 대시보드 실행기"
echo "================================================"

# 가상환경 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 1. 분석 결과 최신화
echo "🧠 최신 데이터를 기반으로 AI 분석 중..."
export PYTHONPATH=$PYTHONPATH:.
python3 src/export_results.py

echo "✅ 준비 완료! 브라우저에서 아래 주소로 접속하세요:"
echo "👉 http://127.0.0.1:8002"
echo "================================================"

# 2. 정적 웹 서버 실행
python3 -m http.server 8002
