#!/bin/bash
set -eu
cd "$(dirname "$0")"
src=../dcf10_amortized_routing
cp "$src/router10.py" "$src/protocol.json" .
mkdir -p deps/deps
for f in adaptive.py experiment.py followup_spec.json spec.json; do cp "$src/deps/$f" deps/; done
for f in followup.py gym.py router.py protocol.json spec.json; do cp "$src/deps/deps/$f" deps/deps/; done
sha256sum -c PRE_RUN_SHA256
