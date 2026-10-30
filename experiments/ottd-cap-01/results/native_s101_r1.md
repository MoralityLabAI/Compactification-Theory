# OTTD-CAP-01: native OpenTTD 15.3 run S101-R1

Date: 2026-10-30 (Brunei); GitHub Actions run [40935557048](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40935557048).

## Provenance and validity

Upstream source `thesophiaxu/openttd-bench` pinned to `fee25ab19837959fd9c9749808b117144b9d1b3e`; actual OpenTTD dedicated server reported version 15.3. Native engine and GameScript were initialized through the upstream `init_game.py` development path on Ubuntu CI. The short scenario derives from upstream `two_towns_rail` seed 101, enabling road+rail, a single player seat, a one-year 400%-speed real-time clock and $2,000,000 starting cash. No model inference or external Prime billing was used. This is a **modified smoke fixture**, not an official leaderboard submission or the competitive ten-year task.

## Directly observed results

- Founded towns Northgate (108,128) and Southbridge (148,128). The live survey selected a straight passenger road from (112,131) to (146,131), with a depot at (129,132). Passenger catchment scans at both stations passed the selector's prebuild threshold.
- **Eight of eight** sequential native execution steps completed with a receipt: straight road; two bus stations; depot; passenger bus; two asynchronous station orders; vehicle start. The upstream order ticket IDs 1 and 2 both resolved successfully.
- The created bus was a **MPS Regal Bus**, engine ID 116, capacity 31 PASS; created vehicle ID 2 with two station orders. Its post-start state was reported as `running`.
- A live company snapshot in August 1950 reported **70 road pieces, 2 stations, 1 road vehicle**, both orders present, $1,984,655 remaining cash and a valid `company-quarterly-v1` accounting record.
- Upstream validated construction cost sum in the controller was **$14,812**. This is a prebuild validation sum, not the final authoritative expenditure ledger.
- At the native snapshot (1950-08-23), cumulative operating profit was **-$358**, with income 0, expenses -358 and **0 monitored PASS deliveries**. The passenger-delivery acceptance condition failed.

## Evidence boundaries

These observations are from the real OpenTTD engine and authenticated API, including the upstream control plane's native grading-snapshot endpoint. The controller's construction receipt verifies one native route startup; it does not establish a connected service, delivered cargo, profitable operations, or a benchmark score. End-of-year terminal scoring was not performed. Source hashes, game logs, plan, receipt and metrics are attached to the run's short-retention GitHub Actions artifact.

## Next diagnostic gate

A follow-up real-game run instruments moving vehicle coordinates and speed, PASS load, station cargo and town production at several game dates. It distinguishes route/depot connectivity from absent passenger catchment or service. A separate candidate design is a perpendicular two-tile depot-access spur to form a T junction on the main road. That branch must pass native validation, the resulting vehicle must visit both stops, and the actual PASS accumulator must rise above zero before declaring the service viable.
