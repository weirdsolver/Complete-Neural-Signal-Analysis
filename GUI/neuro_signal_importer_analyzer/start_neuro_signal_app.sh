#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 run_neuro_signal_app.py "$@"
