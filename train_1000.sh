#!/bin/bash
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=$PYTHONPATH:.
python train_1000.py
