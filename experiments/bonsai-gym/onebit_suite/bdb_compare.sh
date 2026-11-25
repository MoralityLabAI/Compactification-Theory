#!/usr/bin/env bash
# Bonsai 2.5 27B BDB adapters vs the Inkling 2 Mini (Tinker) BDB adapter, one GPU lease at a time, resumable:
#   1  train the 1-bit and full-precision BDB adapters (resuming checkpoints), each through the bench (episodes,
#      development seeds 0-7) and through the 36-row eval split; the bare model through the split
#   2  Inkling 2 Mini base and LoRA-folded on squid5090 through the same bench (adapter-bits
#      experiments/inky-tinker/inky_bdb_arm.sh; its 36-row split scores are in inky-tinker/results/bdb-full-squid)
#   3  the rest of the 1-bit suite (run_suite.sh skips what is done)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GYM=$(cd "$HERE/.." && pwd)
. "$HERE/suite_env.sh"
. "$HERE/suite_lib.sh"
INKY=$(cd "$GYM/../inky-tinker" && pwd)
bdb_split base
train "${CFG_PFX}onebit_bdb" && bdb_arm onebit-r8 "$WW/runs/${CFG_PFX}onebit_bdb/adapter.gguf" \
  && bdb_split solo-onebit-bdb "$WW/runs/${CFG_PFX}onebit_bdb/adapter.gguf"
train "${CFG_PFX}bdb_fp_r8" && bdb_arm fp-r8 "$WW/runs/${CFG_PFX}bdb_fp_r8/adapter.gguf" \
  && bdb_split solo-fp-bdb "$WW/runs/${CFG_PFX}bdb_fp_r8/adapter.gguf"
for arm in base lora-fold; do bash "$INKY/inky_bdb_arm.sh" "$arm"; done
log "BDB COMPARE COMPLETE"
bash "$HERE/run_suite.sh"
