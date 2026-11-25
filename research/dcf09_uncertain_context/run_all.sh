#!/bin/bash
set -eu
cd "$(dirname "$0")"
mkdir -p runs
xargs -P 3 -n 2 bash -c 'python experiment.py --k "$0" --seed "$1" --out "runs/${0}_${1}" > "runs/${0}_${1}.log" 2>&1' < cases.txt
