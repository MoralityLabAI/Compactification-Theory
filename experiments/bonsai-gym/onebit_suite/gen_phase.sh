#!/usr/bin/env bash
# One generation phase of the 1-bit adapter suite, run inside a vramorama lease (vramorama run ... --guard-power 600
# --restart N -- bash gen_phase.sh <phase>). Starts Bonsai 2 27B llama-server with 4 slots, waits for /health,
# runs generate.py jobs (each resumable, so a power-guard restart continues where it stopped), and exits; the
# lease's job object ends the server.
#   base   bare model: eval splits of all four families (greedy), then RFT samples on the math and logic pools
#   cross  all four 1-bit adapters loaded unapplied; each adapter in turn on every family's eval split (greedy)
set -u
PHASE=$1
PORT=${PORT:-18299}
W=/c/Users/patri/Documents/Codex/onebit-suite
WW='C:/Users/patri/Documents/Codex/onebit-suite'
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/c/Users/patri/.venvs/onebit-suite/Scripts/python.exe
ADAPTERS=math,logic,srb,bdb
EXTRA=()
if [ "$PHASE" = cross ]; then
  L=""
  for a in ${ADAPTERS//,/ }; do
    g="$WW/runs/onebit_$a/adapter.gguf"
    [ -s "$g" ] || { echo "missing $g"; exit 2; }
    L="${L:+$L,}$g"
  done
  EXTRA=(--lora "$L" --lora-init-without-apply)
fi
mkdir -p "$W/logs"
/c/Users/patri/tools/llama-prism-b10743/bin/llama-server.exe -m 'C:\Users\patri\models\ternary-bonsai-2-27b\Ternary-Bonsai-2-27B-PTQ1_0.gguf' \
  --alias bonsai-2-27b --host 127.0.0.1 --port "$PORT" -ngl 99 --load-mode none --ctx-size 18432 -np 4 \
  -ctk q8_0 -ctv q8_0 -fa on -b 256 -ub 128 --jinja --reasoning off --reasoning-budget 0 --no-reasoning-preserve \
  --seed 523 --no-warmup --metrics "${EXTRA[@]}" > "$W/logs/server-$PHASE-$(date +%H%M%S).log" 2>&1 &
SPID=$!
for i in $(seq 1 900); do
  if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "server exited before ready"; exit 2; fi
  sleep 2
done
echo "server ready $(date +%T)"
curl -s "http://127.0.0.1:$PORT/lora-adapters"; echo

gen() {  # gen <rows> <out> <adapter> <max_tokens> [extra generate.py args]
  local rows=$1 out=$2 ad=$3 mt=$4; shift 4
  mkdir -p "$(dirname "$out")"
  echo "GEN $(basename "$(dirname "$out")")/$(basename "$out") adapter=$ad $(date +%T)"
  "$PY" -u "$HERE/generate.py" --rows "$rows" --out "$out" --base-url "http://127.0.0.1:$PORT/v1" \
    --adapter "$ad" --adapters "$([ "$PHASE" = cross ] && echo $ADAPTERS)" --max-tokens "$mt" --concurrency 4 "$@" || exit 4
}
mt() { case $1 in srb) echo 1024;; bdb) echo 1024;; *) echo 1024;; esac; }

if [ "$PHASE" = base ]; then
  for fam in srb bdb math logic; do gen "$WW/data/$fam/eval.jsonl" "$WW/gen/base/$fam.jsonl" none "$(mt $fam)"; done
  gen "$WW/data/math/rft_pool.jsonl" "$WW/gen/rft/math.jsonl" none 1024 --temperature 0.7 --top-p 0.95
  gen "$WW/data/logic/rft_pool.jsonl" "$WW/gen/rft/logic.jsonl" none 1024 --temperature 0.7 --top-p 0.95
elif [ "$PHASE" = cross ]; then
  for ad in ${ADAPTERS//,/ }; do
    for fam in srb bdb math logic; do gen "$WW/data/$fam/eval.jsonl" "$WW/gen/$ad/$fam.jsonl" "$ad" "$(mt $fam)"; done
  done
fi
echo "PHASE $PHASE COMPLETE $(date +%T)"
