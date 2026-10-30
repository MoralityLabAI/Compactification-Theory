# OTTD-CAP-01: native OpenTTD-Bench controller

**Evidence:** Offline compiler, mocked upstream API suite, and an actual native OpenTTD 15.3 single-company smoke that delivered 31 passengers and produced +$1,061 operating profit as of 1950-04-16. See [S101-R3 evidence](results/native_s101_r3.md). This is an interim snapshot in a modified one-year game; no official ten-year competitive benchmark result is claimed. Upstream pinned to Sophia Xu's OpenTTD-Bench commit fee25ab19837959fd9c9749808b117144b9d1b3e.

## Supported plans

The native_controller.py compiler accepts a fixed two-stop route in either road or rail mode. Road plans contain contiguous orthogonal segments, bus/truck stops, a road depot and one road vehicle. Rail plans contain directional platforms, explicit OpenTTD from/via/to track segments, a depot, locomotive and one to six wagons. The compiler appends two station orders and starts the vehicle after both tickets resolve.

Inputs use a closed grammar with type, adjacency, station-axis, step-count, spend, wall-clock and latest-start-year limits. It resolves station and vehicle IDs from server responses. Terrain, tile occupancy, station passenger acceptance, rail alignment, engine eligibility, route connectivity and cargo economics require validation in the actual game. The two included coordinate/engine examples are **illustrative** and must be replaced with map-surveyed values.

## Compile offline, without a server

From repository root:

    python experiments/ottd-cap-01/native_controller.py experiments/ottd-cap-01/examples/road_bus.json
    python experiments/ottd-cap-01/native_controller.py experiments/ottd-cap-01/examples/rail_two_stations.json
    python -m unittest discover -s experiments/ottd-cap-01/tests -v

The compile commands read the local plan file and print the exact upstream command sequence. They do not contact the control plane or start a game clock.

## Run against an independently provisioned real-time seat

After provisioning an upstream seat and reviewing route coordinates, engine IDs, map connections and budgets:

    export OPENTTD_API_URL=http://127.0.0.1:8000
    export OPENTTD_API_TOKEN=<your seat token>
    python experiments/ottd-cap-01/native_controller.py /path/to/reviewed-plan.json --execute --allow-clock-start --allow-writes --receipt /path/to/unused-receipt.json

Writes are disabled until both opt-in flags and a unique receipt path are supplied. The runner demands a real-time game mode, permitted transport type, a valid date and available horizon. Each supported construction command gets an upstream test-mode validation immediately before execution. Any loss of validity, spend ceiling or wall budget stops the sequence.

Construction can be asynchronous and is confirmed by independent tile lookups. Vehicle and station identities are re-read after creation. Order changes return an upstream ticket, then the runner polls GET /api/ticket/<id>?timeout=8 and verifies the station appears in the vehicle order list before issuing the next step. It never retries a mutation after a failed confirmation because the server may have committed the action before a connection error.

An exclusive receipt is written before the first live request. Each attempted step includes validation cost, executed command reply, time elapsed, before/after game date, identity evidence and a SHA-256 trace link. Inconclusive confirmations halt the run with a partial receipt. The receipt path cannot be reused for automatic replay. Recovery must start with inspecting the native game state and preparing a newly reviewed plan.

The validated cost sum budgets construction and vehicle spending. It does not report the upstream operating-profit ledger. Real benchmark profit and rankings require authoritative upstream grading snapshots and matched opponent runs. Completion of a route in the API does not prove train motion or profitable cargo delivery.

## Native integration result and next evidence gate

The pinned Linux runner at `.github/workflows/ottd-cap01-native.yml` provisioned a real OpenTTD 15.3 server, surveyed a route, built its connected bus service, and recorded 31 PASS deliveries. The verified company ledger contained $1,170 income, -$109 signed operating expenses and +$1,061 operating profit at 1950-04-16. Its nine-step receipt and source-hashed plan are available in the [passing run artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40937181974). See [S101-R3](results/native_s101_r3.md) for the precise fixture and evidence limits.

A follow-up native CI gate requires `company-quarterly-v1` with a fresh snapshot, valid accounting, correct income/expense arithmetic, an authentic matching passenger-delivery count, and positive income/profit. It additionally recomputes the native plan SHA-256 and step-by-step hash chain with `receipt_verify.py`.

The next experimental step is a **horizon-finalized** one-year grading snapshot, followed by multiple frozen seeds and competitive head-to-head games. A later comparison can evaluate small TRM/HRM planners under matched interface, wall-clock, and inference budgets. The original finite oracle, IRC-02 and Pluto-Gym retain their separate protocols.