# Model, runtime and work-dir settings shared by run_suite.sh, gen_phase.sh and bdb_arm.sh (sourced).
# Default: Ternary Bonsai 2.5 27B (released 2026-11-24) on PrismML llama.cpp prism-b12178 with its built-in
# MTP draft head. Training reads the plain PTQ1_0 file; serving reads the -mtp file (same weights plus the
# draft head), so an adapter trained on one applies to the other. SUITE_MODEL=b2 selects the Bonsai 2 setup.
SUITE_MODEL=${SUITE_MODEL:-b25}
if [ "$SUITE_MODEL" = b25 ]; then
  W=/c/Users/patri/Documents/Codex/onebit-suite-25
  WW='C:/Users/patri/Documents/Codex/onebit-suite-25'
  LLAMA_BIN=/c/Users/patri/tools/llama-prism-b12178/bin
  SERVE_GGUF='C:\Users\patri\models\ternary-bonsai-2.5-27b\Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf'
  SERVE_SHA=e0fc5f3d47b9c926d52677d303b19df356a2376614b3e1cb9973865fc04ca431
  MODEL_NAME=bonsai-2.5-27b
  RUNTIME="llama-prism-b12178 (0.7.0-dev commit 03ec60072) PTQ1_0-mtp pure-GPU q8 KV, MTP draft n-max 3 p-min 0.1"
  SPEC=(--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1)
  CFG_PFX=b25_
else
  W=/c/Users/patri/Documents/Codex/onebit-suite
  WW='C:/Users/patri/Documents/Codex/onebit-suite'
  LLAMA_BIN=/c/Users/patri/tools/llama-prism-b10743/bin
  SERVE_GGUF='C:\Users\patri\models\ternary-bonsai-2-27b\Ternary-Bonsai-2-27B-PTQ1_0.gguf'
  SERVE_SHA=53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3
  MODEL_NAME=bonsai-2-27b
  RUNTIME="llama-prism-b10743 (0.2.0-dev commit adfffbe41) PTQ1_0 pure-GPU q8 KV"
  SPEC=()
  CFG_PFX=
fi

# start_server <port> <n_slots> <ctx> <log> [extra args...]: llama-server in the background, SPID set, waits for
# /health. A server that exits before ready with speculative decoding on is retried once without it (logged).
start_server() {
  local port=$1 np=$2 ctx=$3 logf=$4; shift 4
  local spec=("${SPEC[@]}")
  for attempt in 1 2; do
    "$LLAMA_BIN/llama-server.exe" -m "$SERVE_GGUF" --alias "$MODEL_NAME" --host 127.0.0.1 --port "$port" -ngl 99 \
      --load-mode none --ctx-size "$ctx" -np "$np" -ctk q8_0 -ctv q8_0 -fa on -b 256 -ub 128 --jinja \
      --reasoning off --reasoning-budget 0 --no-reasoning-preserve --min-p 0 --seed 523 --temp 0 --metrics \
      "${spec[@]}" "$@" > "$logf" 2>&1 &
    SPID=$!
    for i in $(seq 1 900); do
      if curl -s "http://127.0.0.1:$port/health" | grep -q '"ok"'; then echo "server ready (spec: ${spec[*]:-off}) $(date +%T)"; return 0; fi
      kill -0 $SPID 2>/dev/null || break
      sleep 2
    done
    kill $SPID 2>/dev/null
    echo "server failed to start (spec: ${spec[*]:-off}):"; tail -15 "$logf"
    [ ${#spec[@]} -eq 0 ] && return 2
    spec=(); RUNTIME="$RUNTIME [MTP off: server refused it]"
  done
  return 2
}

# warm <port>: one short request, so the first measured request is not a cold one
warm() {
  curl -s "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"x","max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"ok"}]}' > /dev/null
}
