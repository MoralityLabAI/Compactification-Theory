# OTTD-CAP-01: transport economics, memory and interface limits

**Status:** Runnable finite synthetic experiment; gated native controller; **first verified live OpenTTD 15.3 passenger delivery and positive operating ledger** in a modified single-player smoke fixture. No model training or official competitive leaderboard evaluation has been performed.

Origin: Sophia Xu's [OpenTTD-Bench](https://github.com/thesophiaxu/openttd-bench), pinned at commit fee25ab19837959fd9c9749808b117144b9d1b3e. Upstream source uses CC0; OpenTTD and downloaded game content have their own licenses. This subfolder changes no upstream files and makes no calls to paid model APIs.

## Run with Python 3.12+ (stdlib only)

From the root of Compactification-Theory:

    python -m unittest discover -s experiments/ottd-cap-01/tests -v
    python -O -m unittest discover -s experiments/ottd-cap-01/tests -v
    python experiments/ottd-cap-01/bench.py --mode sweep --seeds 101,202,303 --include-traces --output /tmp/ottd-cap01.json
    python experiments/ottd-cap-01/bench.py --mode exact --routes 2 --horizon 5 --memory-bits 3

The paired sweep compares a memoryless public-prior greedy policy with scouting policies carrying 0, 3, 6 or 9 retained bits. The same generated routes and latent quality mask are used across arms for each seed. Each episode emits a hash-chain receipt. The optional exact fixture enumerates all latent quality masks and all legal future feedback branches using rational probabilities.

A route requires an investment. It then earns or loses operating income in future months. Surveying consumes one month, and the revealed route quality is visible at exactly one subsequent decision unless the policy retains it. Capital spending lowers cash without entering the operating-profit accumulator. This synthetic environment is a tiny partially observed economy, not the OpenTTD game engine.

## Why operating profit is separate

The [upstream mixed_hogex_realtime_10y task](https://github.com/thesophiaxu/openttd-bench/blob/fee25ab19837959fd9c9749808b117144b9d1b3e/openttd_bench/tasks/mixed_hogex_realtime_10y.py) scores a 10-year real-time game. Its main objectives are operating profit against a 3,000,000 target (60%) and profit rank against rivals (40%). Construction and vehicle purchases are excluded from the operating ledger. The synthetic single-firm oracle optimizes terminal **net cash**, recording operating profit and capex separately. An upstream-style scoring helper tests the exact profit/rank algebra given supplied measurements. Synthetic cash values are never submitted as real scores.

## Live upstream seat adapter

The bridge uses the actual authenticated control-plane API paths: GET /api/state and POST /api/batch. Its command names come from the pinned upstream schema, with an explicit read/write allowlist. Writes are disabled by default, even if a policy requests them.

For a seat that has already been provisioned by the upstream runner:

    OPENTTD_API_URL=http://127.0.0.1:8000 OPENTTD_API_TOKEN=YOUR_SEAT_TOKEN python experiments/ottd-cap-01/bench.py --mode probe

The default probe reads state only. Add --include-map to inspect towns; even read-only action requests can **start the upstream real-time clock**. To use writes, an embedding harness must construct SeatBridge(allow_writes=True) after explicit policy and billing approval. This package includes a bounded public-observation route selector and a native Linux CI launcher; live engine results are documented below.

For existing verified upstream trajectories:

    python experiments/ottd-cap-01/bench.py --mode atif --atif /path/to/seat.atif.json

ATIF inventory reports the source SHA-256, tool activity and upstream-declared usage. Score fields are left unverified until an authoritative upstream end-of-game grading snapshot exists.

## Research ownership

The exact finite Bayesian *value* oracle and synthetic controller comparisons belong to this experiment. The canonical full adaptive future-trace history-equivalence oracle is separately owned by [EvalMath IRC-02](https://github.com/MoralityLabAI/EvalMath/pull/21). [Compactification-Theory IRC-02](../interface-relative-compactification/README.md) owns the cross-environment resource protocol. [Pluto-Gym](../../research/pluto_gym/README.md) owns its synthetic scheduler and game-design curriculum. Native source retains a time-aware plan-to-command translation and a separate, authenticated interim grading snapshot.


## Native controller milestone (October 30, 2026)

The experiment now includes a schema-constrained native road/rail controller in [native_controller.py](native_controller.py), with [execution instructions](NATIVE.md) and illustrative plans under examples/. It captures real station and vehicle IDs, validates construction before each write, confirms asynchronous order tickets and records a stop-on-failure audit receipt.

Both example plans compile fully offline:

    python experiments/ottd-cap-01/native_controller.py experiments/ottd-cap-01/examples/road_bus.json

Live writes require explicit clock and write gates and a separately provisioned upstream seat. Both mocked tests and a real OpenTTD 15.3 integration smoke have passed. No official ten-year competitive benchmark score has been measured.


## First verified native passenger delivery (S101-R3, October 30, 2026)

The [pinned native OpenTTD 15.3 run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40937181974) passed on GitHub Actions Ubuntu. A real game on seed 101 used a deterministic, single-company, one-year, accelerated real-time smoke fixture derived from upstream `two_towns_rail`. The live controller surveyed 480 corridor candidates, built a connected two-stop bus service with a perpendicular depot spur, and observed the bus reach the destination.

At the authenticated native snapshot on 1950-04-16, the cargo accumulator and grading snapshot each reported **31 PASS delivered**. Native `company-quarterly-v1` accounting was fresh and valid: **$1,170 transport revenue, -$109 running expenses, +$1,061 cumulative operating profit**. Both `road.connected` checks passed. Nine native action steps were confirmed; the source-hashed plan and receipt chain were independently recomputed against the saved artifact.

See [the full evidence and limitations](results/native_s101_r3.md). This is **interim native-game evidence**, preceding the 1951 horizon, with no paid model calls or rival companies. The 10-year competitive benchmark and model-to-model comparisons remain to be evaluated. The native CI now also requires a fresh, internally consistent ledger with positive passenger revenue/profit and verifies the receipt chain against its original plan.
