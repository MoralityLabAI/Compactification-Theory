#!/usr/bin/env bash
# Bonsai 2 27B PTQ1_0 decode speed: plain file vs the same weights with Bonsai 2.5's MTP head grafted (mtp_graft.py)
# and MTP speculative decoding on. Server flags match the MIT-13 pilot (run_bonsai2.sh); the first prompt is a warm-up.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRV=/c/Users/patri/tools/llama-prism-b12178/bin/llama-server.exe
MD='C:\Users\patri\models\ternary-bonsai-2-27b'
GYM='C:\Users\patri\OneDrive\Documents\GitHub\EvalMath-mit\gyms\MathematicalInstrumentTransfer'
PROMPTS='C:\Users\patri\Documents\Codex\mit13-work\bonsai2\prompts.jsonl'
OUT=/c/Users/patri/Documents/Codex/bonsai2-mtp-bench
PORT=18214
N=${N:-12}
# ARMS: plain, mtp25graft, or plain-rep (the plain file again: a determinism control for the identity check)
for arm in ${ARMS:-plain mtp25graft}; do
  [ -s "$OUT/$arm.jsonl" ] && [ "$(wc -l < "$OUT/$arm.jsonl")" -ge $((N + 1)) ] && { echo "$arm done"; continue; }
  if [ $arm != mtp25graft ]; then M="$MD\Ternary-Bonsai-2-27B-PTQ1_0.gguf"; EXTRA=()
  else M="$MD\Ternary-Bonsai-2-27B-PTQ1_0-mtp25graft.gguf"; EXTRA=(--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1); fi
  "$SRV" -m "$M" -ngl 99 -c 8192 -np 1 -fa on --jinja --reasoning-format deepseek --reasoning-budget 1024 \
    --min-p 0 --port $PORT --no-webui "${EXTRA[@]}" > "$OUT/server-$arm.log" 2>&1 &
  SP=$!
  for i in $(seq 1 180); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  if ! curl -s http://127.0.0.1:$PORT/health | grep -q ok; then echo "$arm server failed"; tail -20 "$OUT/server-$arm.log"; continue; fi
  python -u "$HERE/bench_client.py" --prompts "$PROMPTS" --gym "$GYM" --port $PORT --arm $arm --n $N --out "$(cygpath -w "$OUT/$arm.jsonl")"
  taskkill //PID $(cat /proc/$SP/winpid) //F > /dev/null 2>&1 || kill $SP
  sleep 5
done
python "$HERE/summarize.py" "$(cygpath -w "$OUT")"
