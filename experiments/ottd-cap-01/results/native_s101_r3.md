# OTTD-CAP-01 native evidence S101-R3: first verified revenue

Date: 2026-10-30 (Asia/Brunei). [Passing native GitHub Actions run #40937181974](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40937181974). [Evidence artifact #12924988406](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40937181974/artifacts/12924988406).

## Source, experiment and evidence scope

- Upstream `thesophiaxu/openttd-bench` detached checkout at SHA `fee25ab19837959fd9c9749808b117144b9d1b3e`; native dedicated server identified as OpenTTD **15.3**. The actual upstream GameScript/API ran on GitHub Actions Ubuntu. World was derived from the pinned `two_towns_rail` task, map seed **101**, founded towns Northgate and Southbridge, one LLMCompany-controlled company with **$2,000,000** starting cash, permitted road and rail modes, no loans, accelerated realtime clock **400%**, and one-year horizon **1950–1951**.
- Controller is deterministic rule-based code with **zero model inference or paid Prime calls**. This custom smoke fixture measures API and game integration. It is not Sophia Xu's official 10-year competitive benchmark, and there is no opponent or leaderboard claim.
- The runner surveyed **480** candidate corridors through public `inspect.*` and validation API responses, yielding four viable candidates. It chose a passenger corridor at **y=131**, endpoints **(110,131)** and **(144,131)**, with a perpendicular road spur **(127,131)** to **(127,132)** and a depot facing the spur at **(127,133)**.

## Native construction and independent checks

- Upstream accepted nine controller actions: main `road.road`, spur `road.road`, two `road.stop`, one `road.depot`, `vehicles.build`, two `orders.append`, and `vehicles.start`. All nine are marked `confirmed` in the native step-by-step receipt.
- Both actual connectivity calls reported `road.connected = true`: trunk-to-spur and spur-to-depot. The company had 72 road pieces, two stations and one road vehicle.
- The actual passenger vehicle was the **MPS Regal Bus** (engine 116, capacity 31 PASS, vehicle ID 2), with orders to station IDs **0** and **1**. By April 16 it had reached **(144,131)** at the second station and was carrying new PASS cargo after its first delivery.
- The saved native receipt's chained SHA-256 value is `410bdc1bcfeab4758111084a7382838ab86ea8ac660c9a2356316b58ddb4748d`; its input plan SHA-256 is `ee977daaca87634bffe0767c37b892191ad6f5bd3bcbf52cdb8132bcf0328d46`. Independently recomputing these hashes from the uploaded native evidence ZIP reproduces both values. The sum of per-step validation costs is **$15,267**, which is a test-mode cost estimate rather than the authoritative expenditure ledger.

## Cargo and financial measurement, 1950-04-16

| Native source field | Actual value |
| --- | ---: |
| `inspect.deliveries` PASS | **31** |
| Grading snapshot `cargo_delivered.PASS` | **31** |
| Cargo destination (town ID 2) | **31 PASS** |
| `accounting.income` | **$1,170** |
| `accounting.expenses` | **-$109** |
| `accounting.operating_profit` | **+$1,061** |
| Cash at snapshot | **$1,985,719** |
| Company value at snapshot | **$2,000,263** |
| Accounting version/valid | `company-quarterly-v1` / true |
| `accounting_snapshot.fresh` / error | true / null |
| Game date / horizon | 1950-04-16 / 1951 |

Real operating profit follows the native **income + signed expenses** ledger, 1170 + (-109) = **1061**, and is separate from construction cost or terminal wealth. The cargo accumulator and grading snapshot agree on delivered passengers. This is an **interim native state**, recorded before the horizon. The engine was still running and its final 1951 profit, rank and terminal benchmark score remain unmeasured.

## Diagnostic chronology

- S101-R1 established native construction success but had 0 PASS deliveries and -$358 operating profit with an inaccessible depot.
- S101-R2 instrumented vehicle location, cargo and town production, showing the bus remaining on its depot tile while passengers remained unserved.
- A new branching road compiler and route selector added a short perpendicular depot spur. The selector also rejects obstructed station sites and validates road placement before live construction.
- S101-R3 is the first real engine run with a connected road spur. The native API subsequently observed the bus leave the depot, reach the destination, and deliver 31 PASS, producing a positive quarterly operating ledger.

## Independent replay and outstanding work

The artifact contains `fixture.json`, `native-plan.json`, `native-receipt.json`, `report.json`, and bootstrap/API/OpenTTD logs, and deliberately excludes seat/auth tokens. An offline integrity check recomputed the plan digest, chained receipt digest, and **$15,267** validation-cost sum from the artifact. Receipt hashing checks internal consistency; trust in the game outcome comes from the live authenticated control-plane snapshot and its accounting validity flags.

Future work: record a horizon-finalized grading snapshot after a full year, run more map seeds with fixed scripted baselines and matched rivals, then compare TRM/HRM planners under identical inference, memory, wall time and action access. Keep the native-passenger delivery result distinct from competitive benchmark parity.
