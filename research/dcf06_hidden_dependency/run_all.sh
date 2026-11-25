#!/bin/bash
set -eu
cd "$(dirname "$0")"
mkdir -p runs
xargs -P 4 -n 3 bash -c 'python gym.py --k "$0" --family "$1" --seed "$2" --out "runs/${0}_${1}_${2}" > "runs/${0}_${1}_${2}.log" 2>&1' < cases.txt
