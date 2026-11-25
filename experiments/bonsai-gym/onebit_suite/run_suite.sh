#!/usr/bin/env bash
# The 1-bit adapter suite end to end, one GPU lease at a time; every step skips itself when its output exists,
# so the script can be rerun after any stop. Model and paths come from suite_env.sh (default Bonsai 2.5 27B).
# Bitcoin DeFi Bench first:
#   1  BDB bare-model arm (the dry run this model's adapters are compared with)
#   2  train the BDB adapters (1-bit, full precision) and run each through the bench; train the SRB adapter
#   3  base generation (gen_phase.sh base): bare-model scores on every family's eval split + RFT samples
#   4  score the base arm; build RFT training rows for math and logic; train those two adapters
#   5  cross generation (gen_phase.sh cross): every 1-bit adapter on every family's eval split
#   6  score every arm; train and score the TRM router
#   7  BDB hill-climb round 1: the 1-bit adapter on 4x the training seeds, through the bench
# Training and generation pause in place while the GPU is power-capped (canary.py / generate.py cap watch),
# so the vramorama power guard is only a long backstop here.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GYM=$(cd "$HERE/.." && pwd)
. "$HERE/suite_env.sh"
. "$HERE/suite_lib.sh"
done_gen() {  # done_gen <arm>: every eval split fully generated for that arm
  local arm=$1
  for fam in $FAMS; do
    [ -s "$W/gen/$arm/$fam.jsonl" ] || return 1
    [ "$(wc -l < "$W/gen/$arm/$fam.jsonl")" -ge "$(wc -l < "$W/data/$fam/eval.jsonl")" ] || return 1
  done
}
gen_phase() {  # gen_phase <base|cross> <done-check>
  local phase=$1 check=$2
  for try in 1 2 3 4 5; do
    $check && return 0
    log "gen $phase (attempt $try)"
    vramorama run --vram 9G --owner "suite-gen-$phase" --guard-power 1800 --restart 20 -- bash "$HERE/gen_phase.sh" "$phase"
    log "gen $phase rc=$?"
  done
  $check || { log "gen $phase incomplete"; return 1; }
}

# 1
bdb_arm base
# 2
train "${CFG_PFX}onebit_bdb" && bdb_arm onebit-r8 "$WW/runs/${CFG_PFX}onebit_bdb/adapter.gguf"
train "${CFG_PFX}bdb_fp_r8" && bdb_arm fp-r8 "$WW/runs/${CFG_PFX}bdb_fp_r8/adapter.gguf"
train "${CFG_PFX}onebit_srb" || exit 6

# 3
base_done() { done_gen base && [ "$(wc -l < "$W/gen/rft/logic.jsonl" 2>/dev/null || echo 0)" -ge "$(wc -l < "$W/data/logic/rft_pool.jsonl")" ]; }
gen_phase base base_done || exit 4

# 4
log "score base, build RFT rows"
"$PY" "$HERE/score_all.py" score --work "$WW" --arm base || exit 5
[ -s "$W/data/math/rft_train.jsonl" ] || "$PY" "$HERE/score_all.py" rft --work "$WW" --family math --val 0 || exit 5
[ -s "$W/data/logic/rft_train.jsonl" ] || "$PY" "$HERE/score_all.py" rft --work "$WW" --family logic --val 24 || exit 5
cat "$W/data/math/gold_train.jsonl" "$W/data/math/rft_train.jsonl" > "$W/data/math/train.jsonl"
cp "$W/data/logic/rft_train.jsonl" "$W/data/logic/train.jsonl"
cp "$W/data/logic/rft_val.jsonl" "$W/data/logic/val.jsonl"
wc -l "$W"/data/*/train.jsonl "$W"/data/*/val.jsonl
train "${CFG_PFX}onebit_math" || exit 6
train "${CFG_PFX}onebit_logic" || exit 6

# 5
cross_done() { done_gen math && done_gen logic && done_gen srb && done_gen bdb; }
gen_phase cross cross_done || exit 4

# 6
log "score arms, router"
for arm in math logic srb bdb; do "$PY" "$HERE/score_all.py" score --work "$WW" --arm "$arm" || exit 7; done
"$TPY" "$HERE/router.py" --data "$WW/data" --scores "$WW/scores" --out "$WW/results/router" || exit 8

# 7
[ "$SUITE_MODEL" = b25 ] && train b25_onebit_bdb_x4 && bdb_arm onebit-r8-x4 "$WW/runs/b25_onebit_bdb_x4/adapter.gguf"
log "SUITE COMPLETE"
