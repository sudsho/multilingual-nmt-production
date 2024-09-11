#!/usr/bin/env bash
# Quick smoke test against a running API.
set -euo pipefail
HOST="${1:-http://localhost:8080}"
curl -sf "$HOST/health" | grep -q '"ok"'
curl -sf -X POST "$HOST/translate" -H 'Content-Type: application/json' \
  -d '{"texts":["Hello, world."],"tgt_lang":"hi"}' | python -c 'import json,sys;j=json.load(sys.stdin);assert j["translations"][0],j'
echo "smoke ok"
