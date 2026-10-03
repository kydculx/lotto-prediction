#!/bin/bash
if [ -n "$LOTTO_VENV" ] && [ -d "$LOTTO_VENV" ]; then
    source "$LOTTO_VENV/bin/activate"
elif [ -d "venv-metal" ]; then
    source venv-metal/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-}:."
python train_1000.py "$@"
