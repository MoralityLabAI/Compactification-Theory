#!/bin/bash
# Base vs LoRA-folded Inkling 2 Mini on squid5090 (patched with SQUID_OVERLAY), greedy, MTP off, nvfp4.
# Arms: base (no overlay), attn (attention merged), full (attention merged + NVFP4 experts of layers 0-19).
# Run inside WSL: bash squid_arms.sh <squid binary> <model dir> <overlay dir> <prompts.jsonl> <out dir>
set -u
BIN=$1; MODEL=$2; OV=$3; PROMPTS=$4; OUT=$5; PORT=${PORT:-18330}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
n=$(grep -c . "$PROMPTS")
for arm in ${ARMS:-base attn full}; do
  [ -s "$OUT/samples_$arm.jsonl" ] && [ "$(grep -c . "$OUT/samples_$arm.jsonl")" -ge "$n" ] && { echo "$arm done"; continue; }
  case $arm in
    base) export SQUID_OVERLAY= ;;
    attn) export SQUID_OVERLAY=$OV/attn.safetensors ;;
    full) export SQUID_OVERLAY=$OV/attn.safetensors,$OV/experts.safetensors ;;
  esac
  echo "=== $arm overlay=[$SQUID_OVERLAY] $(date +%T)"
  "$BIN" serve "$MODEL" --host 127.0.0.1 --port $PORT --mode nvfp4 --mtp 0 --ctx 8192 --temp 0 > "$OUT/server-$arm.log" 2>&1 &
  SP=$!
  for i in $(seq 1 240); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  grep -E "overlay|Fitted|error|panic" "$OUT/server-$arm.log" | head -8
  if ! curl -s http://127.0.0.1:$PORT/health | grep -q ok; then echo "$arm server failed"; tail -20 "$OUT/server-$arm.log"; kill $SP 2>/dev/null; continue; fi
  echo "ready $(date +%T)"
  python3 -u "$HERE/squid_gen.py" --prompts "$PROMPTS" --out "$OUT/samples_$arm.jsonl" --port $PORT
  kill $SP; wait $SP 2>/dev/null; sleep 5
done
echo "ARMS COMPLETE $(date +%T)"
