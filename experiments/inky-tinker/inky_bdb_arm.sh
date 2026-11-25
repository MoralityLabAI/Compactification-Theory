#!/usr/bin/env bash
# Bitcoin DeFi Bench (bonding, dlc, netting; development seeds 0-7; bare chat agent, greedy, thinking off) on
# Inkling 2 Mini served by squid5090 on the RTX 5080, base or with the Tinker BDB LoRA folded in (squid_overlay.py),
# through inky_proxy.py so prompts use the training renderer. Same agent scripts and token budgets as the Bonsai
# arms (bonsai-gym/onebit_suite/bdb_arm.sh). Resumable: a family whose results file exists is skipped.
#   inky_bdb_arm.sh <base|lora-fold>          (runs inky_bdb_job.sh inside one vramorama lease per attempt)
set -u
ARM=$1
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=/c/Users/patri/OneDrive/Documents/GitHub/bitcoin-defi-bench
last="$BENCH/results/v0-netting-inkling-2-mini-squid-t0-$ARM-development.md"
[ -s "$last" ] && { echo "inky arm $ARM already done"; exit 0; }
mkdir -p /c/Users/patri/Documents/Codex/inky-tinker/bench
for try in 1 2 3; do
  echo "=== inky arm $ARM (attempt $try) $(date '+%F %T')"
  vramorama run --vram 15000M --owner "inky-bdb-$ARM" --guard-power 1800 -- bash "$HERE/inky_bdb_job.sh" "$ARM"
  [ -s "$last" ] && exit 0
done
echo "inky arm $ARM incomplete"
exit 1
