# OTTD-CAP-01: S101-R2 native movement and service diagnostic

Source: [GitHub Actions run 40936189353](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40936189353), upstream `thesophiaxu/openttd-bench` `fee25ab19837959fd9c9749808b117144b9d1b3e`, OpenTTD engine 15.3.

Scope: locally provisioned agent-free OpenTTD test fixture, seed 101, one company, 400% accelerated real-time clock, one-year horizon. No paid inference. This is a native control-plane observation, not an official benchmark score.

## Observed failure

- The compiled two-town bus route created both stations, its depot, a bus and two resolved station orders. The vehicle's state was `running`, with reported speed values varying between 3 and 40 at successive checks.
- For the entire sampled period, January 10 to August 19, 1950, the bus's position remained **(129,125)**, which is also the declared depot tile. Its PASS cargo load remained zero.
- The bus stops were at (112,124) and (147,124). The upstream passenger acceptance sums were 61 and 82; potential production sums were 10 and 12. Two stationary stop cargo records remained empty.
- The native town status showed Northgate PASS production 486 and Southbridge 460 in the last reported month, with **0 supplied** in both towns. The upstream seat's PASS delivery accumulator remained zero.
- The real-time vehicle operating ledger recorded cumulative losses while the route remained unserved; the no-delivery gate rejected the trial.

## Diagnosis and next test

Because the vehicle never left its depot tile, the `running` vehicle state alone provides insufficient evidence of a usable route. A likely mechanical cause is the depot facing an east-west road segment without a perpendicular north-south access connection. This is a **diagnosis awaiting a road-connected check**, not an established explanation from this trace.

The next code revision replaces a depot immediately adjacent to the trunk road with a two-segment connected road network: a straight trunk and a one-tile perpendicular spur, with the depot facing the spur. It checks both `road.connected` edges using the actual upstream API before waiting for cargo revenue. The new success condition additionally requires actual passenger deliveries in the native accumulator.

All recorded values are from native engine/GameScript/ledger observations in the linked CI artifact. The artifact has limited retention, so this note preserves the falsifying observations and the proposed discriminating check.
