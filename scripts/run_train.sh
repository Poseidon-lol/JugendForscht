#!/usr/bin/env bash
set -euo pipefail
python -m src.main train-surrogate --config configs/train_conf.yaml "$@"
