#!/usr/bin/env bash
# One generation phase of the 1-bit adapter suite, run inside a vramorama lease (vramorama run ... -- bash
# gen_phase.sh <phase>). Starts the suite model's llama-server with 4 slots, waits for /health, runs generate.py
# jobs (each resumable, so a restart continues where it stopped), and exits; the lease's job object ends the server.
#   base   bare model: eval splits of all four families (greedy), then RFT samples on the math and logic pools
#   cross  all four 1-bit adapters loaded unapplied; each adapter in turn on every family's eval split (greedy)
set -u
PHASE=$1
PORT=${PORT:-18299}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/suite_env.sh"
PY=/c/Users/patri/.venvs/onebit-suite-sys/Scripts/python.exe
ADAPTERS=math,logic,srb,bdb
EXTRA=()
if [ "$PHASE" = cross ]; then
  L=""
  for a in ${ADAPTERS//,/ }; do
    g="$WW/runs/${CFG_PFX}onebit_$a/adapter.gguf"
    [ -s "$g" ] || { echo "missing $g"; exit 2; }
    L="${L:+$L,}$g"
  done
  EXTRA=(--lora "$L" --lora-init-without-apply)
fi
mkdir -p "$W/logs"
start_server "$PORT" 4 18432 "$W/logs/server-$PHASE-$(date +%H%M%S).log" "${EXTRA[@]}" || exit 2
warm "$PORT"
curl -s "http://127.0.0.1:$PORT/lora-adapters"; echo

gen() {  # gen <rows> <out> <adapter> <max_tokens> [extra generate.py args]
  local rows=$1 out=$2 ad=$3 mt=$4; shift 4
  mkdir -p "$(dirname "$out")"
  echo "GEN $(basename "$(dirname "$out")")/$(basename "$out") adapter=$ad $(date +%T)"
  "$PY" -u "$HERE/generate.py" --rows "$rows" --out "$out" --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL_NAME" \
    --adapter "$ad" --adapters "$([ "$PHASE" = cross ] && echo $ADAPTERS)" --max-tokens "$mt" --concurrency 4 "$@" || exit 4
}

if [ "$PHASE" = base ]; then
  for fam in srb bdb math logic; do gen "$WW/data/$fam/eval.jsonl" "$WW/gen/base/$fam.jsonl" none 1024; done
  gen "$WW/data/math/rft_pool.jsonl" "$WW/gen/rft/math.jsonl" none 1024 --temperature 0.7 --top-p 0.95
  gen "$WW/data/logic/rft_pool.jsonl" "$WW/gen/rft/logic.jsonl" none 1024 --temperature 0.7 --top-p 0.95
elif [ "$PHASE" = cross ]; then
  for ad in ${ADAPTERS//,/ }; do
    for fam in srb bdb math logic; do gen "$WW/data/$fam/eval.jsonl" "$WW/gen/$ad/$fam.jsonl" "$ad" 1024; done
  done
fi
echo "PHASE $PHASE COMPLETE $(date +%T)"
