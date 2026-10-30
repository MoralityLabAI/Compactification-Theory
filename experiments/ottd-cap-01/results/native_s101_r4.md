# OTTD-CAP-01 S101-R4: native replay with ledger and receipt acceptance gates

Date: 2026-10-30 (Asia/Brunei). [Passed native GitHub Actions run #40938136540](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40938136540); [sanitized evidence artifact #12925682130](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40938136540/artifacts/12925682130).

## Repeated independently provisioned native test

The test used the pinned Sophia Xu OpenTTD-Bench source `fee25ab19837959fd9c9749808b117144b9d1b3e`, actual OpenTTD dedicated server 15.3 and authenticated upstream control plane, one company, two-town seed 101, one-year 400%-speed realtime fixture and no model inference. The bridge surveyed public route geometry and engine IDs and executed its chosen road/spur/depot/stations/bus/orders plan. Both native `road.connected` edges passed.

The strengthened acceptance logic requires the native cargo accumulator to report PASS deliveries and an independently obtained admin grading snapshot with fresh and valid `company-quarterly-v1` accounting. It verifies ledger arithmetic (income + signed expenses = cumulative operating profit), the company's own operating-profit field, the native delivered PASS count and positive income/profit.

| Field | Native result |
| --- | ---: |
| Native game date (interim) | **1950-04-08** |
| Verified PASS deliveries | **31** |
| `accounting.income` | **$1,116** |
| `accounting.expenses` | **-$110** |
| `accounting.operating_profit` | **+$1,006** |
| Source ledger | `company-quarterly-v1`, fresh + valid |
| Road-connected tests | Both true |
| Confirmed native action steps | **9/9** |
| Controller validation cost estimate | **$14,992** |
| Replayed receipt status | Complete and valid |

After live native execution, the CI invoked `receipt_verify.py` against `native-receipt.json` and `native-plan.json`. The independent verifier accepted nine confirmed steps with nine attempted writes and recomputed the exact supplied SHA-256 chain and plan hash:

- Receipt trace SHA-256: `c469f1260f0e363c8d4647eb41d8f10a591131cb0f34af48715d7679ecb12dd0`
- Plan SHA-256: `5d5d5854a5e708bee9b6a9ee298942988928ba36c496f97dc77ce5d28cff5adb`

## Comparison with prior native success

S101-R3 ([run #40937181974](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/40937181974)) also delivered 31 PASS, measured **+$1,061** operating profit on **1950-04-16**, and recorded $15,267 as the controller's validation sum. The two native test runs share the same map seed and software pin, while action timing and the selected corridor vary with the game's continuously running clock and real-time validation responses. These are **distinct observed native trajectories**, so differences in interim date and cash/operating-profit readings are part of the evaluation. The results constitute two successful short native integrations, not a statistical model-performance estimate.

## Interpretation

The real engine has now supplied independent evidence of action/connection correctness, cargo delivery and consistent positive quarterly operating earnings in two separate runs. The evidence comes from a shortened one-year solo fixture, observed before its 1951 terminal date. Official ten-year multi-agent profit, rank and leaderboard scores are unmeasured.

The next scientific gate should freeze action time and map-survey budgets, record horizon-finalized balances, and evaluate multiple distinct seeds and controlled opponent companies. Small learned controllers and resource-budgeted LLM planners can then be compared against the identical observation/action interface.
