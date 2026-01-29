#!/bin/bash
# Lotto AI Console Predictor (Mac Double-click)

cd "$(dirname "$0")"

echo "================================================"
echo "🎯 로또 AI 콘솔 분석기"
echo "================================================"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH=$PYTHONPATH:.
python3 main.py --sets 5

echo ""
echo "================================================"
echo "분석이 완료되었습니다. 창을 닫으려면 아무 키나 누르세요."
read -n 1
