#!/bin/bash
# Lotto AI Server Runner

# 현재 스크립트 위치로 이동
cd "$(dirname "$0")"

echo "------------------------------------------------"
echo "🚀 로또 AI 프리미엄 서버를 시작합니다..."
echo "------------------------------------------------"

# 가상환경 활성화 (venv 폴더가 있는 경우)
if [ -d "venv" ]; then
    echo "📦 가상환경(venv) 활성화 중..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "📦 가상환경(.venv) 활성화 중..."
    source .venv/bin/activate
fi

# 필수 라이브러리 설치 확인 (선택 사항)
# echo "📚 종속성 확인 중..."
# pip install -r requirements.txt

# 1. 분석 결과 최신화 (JSON 생성)
echo "🧠 AI 분석 엔진 실행 중..."
export PYTHONPATH=$PYTHONPATH:.
python3 src/export_results.py

# 2. 정적 웹 서버 실행 (GitHub Pages 환경 시뮬레이션)
echo ""
echo "🌐 서버 실행 중: http://127.0.0.1:8002"
echo "🌐 이 환경은 GitHub Pages와 동일한 정적 호스팅 방식입니다."
echo "------------------------------------------------"
python3 -m http.server 8002
