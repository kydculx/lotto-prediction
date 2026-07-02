#!/bin/bash
# Lotto AI Console Predictor

cd "$(dirname "$0")"

echo "================================================"
echo "🎯 로또 AI 콘솔 분석기"
echo "================================================"

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=$PYTHONPATH:.
python3 main.py --sets 10
