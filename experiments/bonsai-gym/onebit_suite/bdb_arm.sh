#!/usr/bin/env bash
# One Bitcoin DeFi Bench arm inside a vramorama lease: start the suite model's llama-server (optionally with one
# LoRA applied), then run families 5, 6, 10 (bonding, dlc, netting) on development seeds 0-7 with the bare chat
# agent, greedy, thinking off -- the same agent scripts and flags as the Bonsai 2 dry run. A family whose results
# file exists is skipped; the agent scripts replay their reply logs, so a stopped arm resumes.
#   bdb_arm.sh <arm-tag> <port> [adapter.gguf]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/suite_env.sh"
ARM=$1; PORT=$2; LORA=${3:-}
NODE=/c/Users/patri/AppData/Local/Programs/nodejs/node.exe
BENCH=/c/Users/patri/OneDrive/Documents/GitHub/bitcoin-defi-bench
EXTRA=()
if [ -n "$LORA" ]; then
  EXTRA=(--lora "$(cygpath -w "$LORA")")
  RUNTIME="$RUNTIME + lora $(sha256sum "$LORA" | cut -c1-16)"
fi
mkdir -p "$W/logs"
start_server "$PORT" 1 16384 "$W/logs/server-bdb-$ARM-$(date +%H%M%S).log" "${EXTRA[@]}" || exit 2
warm "$PORT"
cd "$BENCH"
for spec in "bonding:run_bonding_agent.ts:768" "dlc:run_dlc_agent.ts:768" "netting:run_netting_agent.ts:1024"; do
  IFS=: read fam script mt <<<"$spec"
  out="v0-$fam-$MODEL_NAME-nothink-t0-$ARM-development"
  if [ -s "results/$out.md" ]; then echo "SKIP $out"; continue; fi
  echo "START $out $(date +%T)"
  "$NODE" --experimental-strip-types "scripts/$script" --block development \
    --name "$MODEL_NAME-nothink-t0-$ARM" --out "$out" --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL_NAME" --model-sha256 "$SERVE_SHA" --runtime "$RUNTIME" \
    --temperature 0 --top-p 1 --top-k 1 --seed 523 --thinking false --max-tokens "$mt" \
    > "$W/logs/$out.log" 2>&1
  echo "DONE $out rc=$? $(date +%T)"
done
echo "ARM $ARM COMPLETE"
