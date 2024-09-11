#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/finetune.yaml}"
accelerate launch -m src.train --config "$CONFIG"
