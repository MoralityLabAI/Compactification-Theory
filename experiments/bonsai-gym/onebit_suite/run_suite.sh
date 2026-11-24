#!/usr/bin/env bash
# The 1-bit adapter suite end to end, one GPU lease at a time; every step skips itself when its output exists,
# so the script can be rerun after any stop.
#   1  base generation (gen_phase.sh base): bare-model scores on every family's eval split + RFT samples
#   2  score the base arm; build RFT training rows for math and logic
#   3  train the four 1-bit adapters (canary.py, configs/onebit_<family>.json)
#   4  cross generation (gen_phase.sh cross): every adapter on every family's eval split
#   5  score every arm; train and score the TRM router
# WAIT_FOR: a command-line substring of another job of ours to wait for before the first lease (one queued lease).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GYM=$(cd "$HERE/.." && pwd)
W=/c/Users/patri/Documents/Codex/onebit-suite
WW='C:/Users/patri/Documents/Codex/onebit-suite'
PY=/c/Users/patri/.venvs/onebit-suite/Scripts/python.exe
TPY=/c/Users/patri/.venvs/bluebeam/Scripts/python.exe
FAMS="srb bdb math logic"
log() { echo "=== $* $(date '+%F %T')"; }

if [ -n "${WAIT_FOR:-}" ]; then
  log "waiting for jobs matching '$WAIT_FOR' to end"
  while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*$WAIT_FOR*' -and \$_.Name -eq 'bash.exe' }) { exit 0 } else { exit 1 }"; do
    sleep 120
  done
  log "wait over"
fi

done_gen() {  # done_gen <arm>: every eval split fully generated for that arm
  local arm=$1
  for fam in $FAMS; do
    [ -s "$W/gen/$arm/$fam.jsonl" ] || return 1
    [ "$(wc -l < "$W/gen/$arm/$fam.jsonl")" -ge "$(wc -l < "$W/data/$fam/eval.jsonl")" ] || return 1
  done
}

# 1
base_done() { done_gen base && [ "$(wc -l < "$W/gen/rft/logic.jsonl" 2>/dev/null || echo 0)" -ge "$(wc -l < "$W/data/logic/rft_pool.jsonl")" ]; }
for try in 1 2 3 4 5; do
  base_done && break
  log "1 base generation (attempt $try)"
  vramorama run --vram 7800M --owner onebit-gen-base --guard-power 600 --restart 30 -- bash "$HERE/gen_phase.sh" base
  log "1 rc=$?"
done
base_done || { log "base generation incomplete after 5 attempts"; exit 4; }

# 2
log "2 score base, build RFT rows"
"$PY" "$HERE/score_all.py" score --work "$WW" --arm base || exit 5
[ -s "$W/data/math/rft_train.jsonl" ] || "$PY" "$HERE/score_all.py" rft --work "$WW" --family math --val 0 || exit 5
[ -s "$W/data/logic/rft_train.jsonl" ] || "$PY" "$HERE/score_all.py" rft --work "$WW" --family logic --val 24 || exit 5
cat "$W/data/math/gold_train.jsonl" "$W/data/math/rft_train.jsonl" > "$W/data/math/train.jsonl"
cp "$W/data/logic/rft_train.jsonl" "$W/data/logic/train.jsonl"
cp "$W/data/logic/rft_val.jsonl" "$W/data/logic/val.jsonl"
wc -l "$W"/data/*/train.jsonl "$W"/data/*/val.jsonl

# 3
for fam in $FAMS; do
  if [ -s "$W/runs/onebit_$fam/result.json" ]; then log "3 $fam already trained"; continue; fi
  log "3 train onebit_$fam"
  (cd "$GYM" && PYTHONIOENCODING=utf-8 vramorama run --vram 13G --owner "onebit-train-$fam" --guard-power 120 --restart 40 -- \
     "$TPY" -u scripts/canary.py --config "configs/onebit_$fam.json")
  log "3 $fam rc=$?"
  [ -s "$W/runs/onebit_$fam/adapter.gguf" ] || { log "no adapter for $fam; stopping"; exit 6; }
done

# 4
cross_done() { done_gen math && done_gen logic && done_gen srb && done_gen bdb; }
for try in 1 2 3 4 5; do
  cross_done && break
  log "4 cross generation (attempt $try)"
  vramorama run --vram 8600M --owner onebit-gen-cross --guard-power 600 --restart 40 -- bash "$HERE/gen_phase.sh" cross
  log "4 rc=$?"
done
cross_done || { log "cross generation incomplete after 5 attempts"; exit 4; }

# 5
log "5 score arms, router"
for arm in math logic srb bdb; do "$PY" "$HERE/score_all.py" score --work "$WW" --arm "$arm" || exit 7; done
"$TPY" "$HERE/router.py" --data "$WW/data" --scores "$WW/scores" --out "$WW/results/router" || exit 8
log "SUITE COMPLETE"
