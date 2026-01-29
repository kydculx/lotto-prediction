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

echo "✅ 서버가 가동되었습니다."
echo "✅ 브라우저에서 아래 주소로 접속하세요:"
echo "👉 http://127.0.0.1:5001"
echo "================================================"

python app.py
