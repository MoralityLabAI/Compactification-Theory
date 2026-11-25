#!/usr/bin/env bash
# The 36-row BDB eval split (data/bdb/eval.jsonl, seeds 7000-7003) for one arm inside a vramorama lease: start the
# suite model's llama-server (optionally with one LoRA applied), greedy-generate every row into gen/<arm>/bdb.jsonl
# (resumable), exit. Scored afterwards by score_all.py (bitcoin-defi-bench scripts/score_sft_replies.ts), the same
# per-item scorer as the Inkling 2 Mini adapter's eval (adapter-bits experiments/inky-tinker).
#   bdb_split.sh <arm> <port> [adapter.gguf]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/suite_env.sh"
ARM=$1; PORT=$2; LORA=${3:-}
PY=/c/Users/patri/.venvs/onebit-suite-sys/Scripts/python.exe
EXTRA=()
[ -n "$LORA" ] && EXTRA=(--lora "$(cygpath -w "$LORA")")
mkdir -p "$W/logs" "$W/gen/$ARM"
start_server "$PORT" 4 18432 "$W/logs/server-split-$ARM-$(date +%H%M%S).log" "${EXTRA[@]}" || exit 2
warm "$PORT"
"$PY" -u "$HERE/generate.py" --rows "$WW/data/bdb/eval.jsonl" --out "$WW/gen/$ARM/bdb.jsonl" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL_NAME" --adapter none --max-tokens 1024 --concurrency 4 || exit 4
echo "SPLIT $ARM COMPLETE $(date +%T)"
