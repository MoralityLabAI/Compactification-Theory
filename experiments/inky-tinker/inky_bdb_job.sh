#!/usr/bin/env bash
# One attempt of inky_bdb_arm.sh inside a vramorama lease: bring up squid + proxy in WSL, run the three bench
# families against the proxy, stop the WSL side.
#   inky_bdb_job.sh <base|lora-fold>
set -u
ARM=$1
NODE=/c/Users/patri/AppData/Local/Programs/nodejs/node.exe
BENCH=/c/Users/patri/OneDrive/Documents/GitHub/bitcoin-defi-bench
LOGD=/c/Users/patri/Documents/Codex/inky-tinker/bench
NAME=inkling-2-mini-squid-t0-$ARM
WSLUP=/mnt/c/Users/patri/OneDrive/Documents/GitHub/adapter-bits/experiments/inky-tinker/inky_up.sh
stop_wsl() { MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- pkill -f inky_up.sh; MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- pkill -f squid5080-overlay/bin/squid5090; MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- pkill -f inky_proxy.py; }
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash "$WSLUP" "$ARM" &
WP=$!
for i in $(seq 1 240); do
  curl -s http://127.0.0.1:18330/health | grep -q ok && curl -s http://127.0.0.1:18331/health | grep -q ok && break
  kill -0 $WP 2>/dev/null || break
  sleep 5
done
if ! curl -s http://127.0.0.1:18331/health | grep -q ok || ! curl -s http://127.0.0.1:18330/health | grep -q ok; then
  echo "inky $ARM: squid or proxy not up"; stop_wsl; exit 2
fi
echo "inky $ARM up $(date +%T)"
RUNTIME="squid5090 local 149a746 (port 450e607 + SQUID_OVERLAY), nvfp4, MTP off, ctx 16384, tml_v0 proxy effort 0"
[ "$ARM" = lora-fold ] && RUNTIME="$RUNTIME + Tinker LoRA inky-bdb-full folded (squid_overlay.py seed 0)"
cd "$BENCH"
for spec in bonding:run_bonding_agent.ts:768 dlc:run_dlc_agent.ts:768 netting:run_netting_agent.ts:1024; do
  fam=${spec%%:*}; rest=${spec#*:}; script=${rest%%:*}; mt=${rest#*:}
  out="v0-$fam-$NAME-development"
  if [ -s "results/$out.md" ]; then echo "SKIP $out"; continue; fi
  echo "START $out $(date +%T)"
  "$NODE" --experimental-strip-types "scripts/$script" --block development \
    --name "$NAME" --out "$out" --base-url "http://127.0.0.1:18331/v1" \
    --model inkling-2-mini --model-sha256 65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1 \
    --runtime "$RUNTIME" --temperature 0 --top-p 1 --top-k 1 --seed 523 --thinking false --max-tokens "$mt" \
    > "$LOGD/$out.log" 2>&1
  echo "DONE $out rc=$? $(date +%T)"
done
stop_wsl
wait $WP 2>/dev/null
echo "INKY $ARM COMPLETE $(date +%T)"
