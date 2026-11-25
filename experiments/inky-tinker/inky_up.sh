#!/bin/bash
# WSL side of inky_bdb_arm.sh: squid5090 (overlay build) with or without the LoRA fold, plus inky_proxy.py.
#   inky_up.sh <base|lora-fold>     (stopped from Windows with: wsl.exe ... pkill -f inky_up.sh)
set -u
ARM=$1
HERE=$(cd "$(dirname "$0")" && pwd)
OV=/mnt/c/Users/patri/Documents/Codex/inky-tinker/squid-overlay
LOGD=/mnt/c/Users/patri/Documents/Codex/inky-tinker/bench
mkdir -p "$LOGD"
case $ARM in
  base) export SQUID_OVERLAY= ;;
  lora-fold) export SQUID_OVERLAY=$OV/attn.safetensors,$OV/experts.safetensors ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
/root/squid5080-overlay/bin/squid5090 serve /root/models/inkling-2-mini --host 127.0.0.1 --port 18330 \
  --mode nvfp4 --mtp 0 --ctx 16384 --temp 0 > "$LOGD/squid-$ARM.log" 2>&1 &
SP=$!
(cd /root/inky-tinker && exec venv/bin/python -u "$HERE/inky_proxy.py" --squid-port 18330 --port 18331) > "$LOGD/proxy-$ARM.log" 2>&1 &
PP=$!
trap 'kill $SP $PP 2>/dev/null' EXIT TERM INT
wait $SP
