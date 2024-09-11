#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:-outputs/finetune-en-hi}"
SRC="${2:-en}"
TGT="${3:-hi}"
INPUT="${4:-data/processed/en_hi.val.jsonl}"
python -m src.eval_runner --ckpt "$CKPT" --src "$SRC" --tgt "$TGT" --input "$INPUT"
