#!/usr/bin/env bash
# Inference-only part of the suite, for when the card has room for a 9 GB llama-server but not for training:
# the 36-row BDB split of the bare model, then gen_phase base (bare-model eval splits of all four families and the
# math/logic RFT samples), then the base arm's scores. Resumable; run_suite.sh later skips what is done.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GYM=$(cd "$HERE/.." && pwd)
. "$HERE/suite_env.sh"
. "$HERE/suite_lib.sh"
bdb_split base
for try in 1 2 3 4 5; do
  log "gen base (attempt $try)"
  vramorama run --vram 9G --owner suite-gen-base --guard-power 1800 --restart 20 -- bash "$HERE/gen_phase.sh" base
  log "gen base rc=$?"
  [ "$(grep -c . "$W/gen/rft/logic.jsonl" 2>/dev/null || echo 0)" -ge "$(grep -c . "$W/data/logic/rft_pool.jsonl")" ] && break
done
"$PY" "$HERE/score_all.py" score --work "$WW" --arm base
log "EVALS ONLY COMPLETE"
