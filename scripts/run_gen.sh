#!/usr/bin/env bash
set -euo pipefail
python -m src.main train-generator --config configs/gen_conf.yaml "$@"
