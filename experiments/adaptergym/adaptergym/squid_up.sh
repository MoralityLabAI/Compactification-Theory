#!/bin/bash
# WSL side of the gym's squid backend: squid5090 serve with SQUID_OVERLAY, plus the tml_v0 proxy in front of it.
#   squid_up.sh <squid bin> <model dir> <port> <proxy port> <ctx> <overlays comma list | -> <proxy.py> <proxy python> <effort>
# Stopped from Windows with pkill (see backends.Squid.stop).
set -u
BIN=$1; MODEL=$2; PORT=$3; PPORT=$4; CTX=$5; OV=$6; PROXY=$7; PPY=$8; EFFORT=$9
[ "$OV" = "-" ] && OV=
export SQUID_OVERLAY=$OV
"$BIN" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --mode nvfp4 --mtp 0 --ctx "$CTX" --temp 0 &
SP=$!
(cd "$(dirname "$PPY")/.." && exec "$PPY" -u "$PROXY" --squid-port "$PORT" --port "$PPORT" --effort "$EFFORT") &
PP=$!
trap 'kill $SP $PP 2>/dev/null' EXIT TERM INT
wait $SP
