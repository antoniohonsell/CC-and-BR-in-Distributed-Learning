#!/usr/bin/env bash
set -euo pipefail


KPS="${1:-0.25}"
BYZ="${2:-1}"

mkdir -p data outputs/train outputs/test
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# rosdhb 
python -m ccbr.train.cifar10 --kps "$KPS" --lr 0.7  --byz "$BYZ" --algo rosdhb

# byz_dasha_page 
python -m ccbr.train.cifar10 --kps "$KPS" --lr 0.03 --byz "$BYZ" --algo byz_dasha_page