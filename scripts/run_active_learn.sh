#!/usr/bin/env bash
set -euo pipefail
python -m src.main active-loop --config configs/active_learn.yaml "$@"
