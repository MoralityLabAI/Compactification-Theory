#!/usr/bin/env bash
# Shared helpers for run_suite.sh and bdb_compare.sh (source after suite_env.sh; needs HERE and GYM).
BENCH=/c/Users/patri/OneDrive/Documents/GitHub/bitcoin-defi-bench
PY=/c/Users/patri/.venvs/onebit-suite/Scripts/python.exe
TPY=/c/Users/patri/.venvs/bluebeam/Scripts/python.exe
FAMS="srb bdb math logic"
log() { echo "=== $* $(date '+%F %T')"; }
log "suite model $SUITE_MODEL ($MODEL_NAME), work dir $W"

out_dir() { "$TPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['out_dir'])" "$GYM/configs/$1.json"; }
train() {  # train <config name>
  local cfg=$1 od
  od=$(out_dir "$cfg")
  if [ -s "$od/result.json" ]; then log "train $cfg already done"; return 0; fi
  for try in 1 2 3; do
    log "train $cfg (attempt $try)"
    (cd "$GYM" && PYTHONIOENCODING=utf-8 vramorama run --vram 15G --owner "suite-train-$cfg" --guard-power 1800 --restart 20 -- \
       "$TPY" -u scripts/canary.py --config "configs/$cfg.json")
    log "train $cfg rc=$?"
    [ -s "$od/result.json" ] && return 0
  done
  log "train $cfg failed"; return 1
}
bdb_arm() {  # bdb_arm <arm tag> [adapter.gguf]
  local arm=$1 gguf=${2:-}
  local last="$BENCH/results/v0-netting-$MODEL_NAME-nothink-t0-$arm-development.md"
  [ -s "$last" ] && { log "bdb arm $arm already done"; return 0; }
  for try in 1 2 3; do
    log "bdb arm $arm (attempt $try)"
    vramorama run --vram 8G --owner "suite-bdb-$arm" --guard-power 1800 --restart 20 -- bash "$HERE/bdb_arm.sh" "$arm" 18298 $gguf
    [ -s "$last" ] && return 0
  done
  log "bdb arm $arm incomplete"; return 1
}
bdb_split() {  # bdb_split <arm> [adapter.gguf]: 36-row BDB eval split, then score it
  local arm=$1 gguf=${2:-}
  local n; n=$(grep -c . "$W/data/bdb/eval.jsonl")
  if [ "$(grep -c . "$W/gen/$arm/bdb.jsonl" 2>/dev/null || echo 0)" -lt "$n" ]; then
    for try in 1 2 3; do
      log "bdb split $arm (attempt $try)"
      vramorama run --vram 9G --owner "suite-split-$arm" --guard-power 1800 --restart 20 -- bash "$HERE/bdb_split.sh" "$arm" 18297 $gguf
      [ "$(grep -c . "$W/gen/$arm/bdb.jsonl" 2>/dev/null || echo 0)" -ge "$n" ] && break
    done
  fi
  "$PY" "$HERE/score_all.py" score --work "$WW" --arm "$arm"
}
