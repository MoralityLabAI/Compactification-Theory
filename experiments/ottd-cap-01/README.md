# OTTD-CAP-01: transport economics, memory and interface limits

**Status:** Runnable synthetic experiment plus a schema-scoped adapter for real OpenTTD-Bench seats. Native game runs, model training and leaderboard scores have not been performed by this package.

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

The default probe reads state only. Add --include-map to inspect towns; even read-only action requests can **start the upstream real-time clock**. To use writes, an embedding harness must construct SeatBridge(allow_writes=True) after explicit policy and billing approval. This package has no automatic command planner, seat launcher or real-game action results.

For existing verified upstream trajectories:

    python experiments/ottd-cap-01/bench.py --mode atif --atif /path/to/seat.atif.json

ATIF inventory reports the source SHA-256, tool activity and upstream-declared usage. Score fields are left unverified until an authoritative upstream end-of-game grading snapshot exists.

## Research ownership

The exact finite Bayesian *value* oracle and synthetic controller comparisons belong to this experiment. The canonical full adaptive future-trace history-equivalence oracle is separately owned by [EvalMath IRC-02](https://github.com/MoralityLabAI/EvalMath/pull/21). [Compactification-Theory IRC-02](../interface-relative-compactification/README.md) owns the cross-environment resource protocol. [Pluto-Gym](../../research/pluto_gym/README.md) owns its synthetic scheduler and game-design curriculum. Native controller work will first need a tested, time-aware translation from route plans into stations, tracks, vehicles and orders, with a separate receipt of actual OpenTTD grading.


## Native controller milestone (October 30, 2026)

The experiment now includes a schema-constrained native road/rail controller in [native_controller.py](native_controller.py), with [execution instructions](NATIVE.md) and illustrative plans under examples/. It captures real station and vehicle IDs, validates construction before each write, confirms asynchronous order tickets and records a stop-on-failure audit receipt.

Both example plans compile fully offline:

    python experiments/ottd-cap-01/native_controller.py experiments/ottd-cap-01/examples/road_bus.json

Live writes require explicit clock and write gates and a separately provisioned upstream seat. The mocked API suite has been executed; live OpenTTD and native benchmark scoring remain unverified.
