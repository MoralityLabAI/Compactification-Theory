# DISTILL-07 / research student and native certificate reuse

Tested commit: `9b88d32f4fdd78a604856dbbd7ac8896f62fbf29`. [Student workflow](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46035484987) and [full offline workflow](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46035485070) succeeded. Seven targeted tests and 53 complete Distill tests passed. The hosted job printed the frozen aggregate receipts; per-world rows and local fitted checkpoints were not published. `DISTILL_07_HOSTED_SUMMARY.json` preserves the printed aggregates and receipt identifiers.

## Student

Six CPU-only fits (two initializations per arm), 380 selected train states and 127 calibration states from root-disjoint public programs; 6,840 updates per fit, 41,040 total. Training process CPU totaled 19.363 s. 188,236 world transitions were replayed across all arms, caps, and initializations. Policy-only process CPU totaled 13.332 s across the comparison. These are operation-specific timings and do not include oracle label generation, environment execution or setup.

Priced utility subtracts one utility unit per 4,096 metered logical work units. Both caps are per trajectory. The 25 ms CPU envelope was checked descriptively with zero overruns; measured CPU time varies, so this is not a preemptive or strictly equal CPU-time comparison.

| Policy | Ordinary 8,192 | Shifted 8,192 | Ordinary 32,768 | Shifted 32,768 |
|---|---:|---:|---:|---:|
| reuse_first | 26.557 | 25.327 | 26.557 | 25.327 |
| gated | 26.557 | 25.327 | 26.557 | 25.327 |
| direct_first | 22.812 | 21.661 | 22.812 | 21.661 |
| myopic | 5.000 | 5.000 | 17.796 | 16.905 |
| recurrent | 4.456 | 4.309 | 11.362 | 13.198 |
| flat | 1.057 | -1.184 | 8.943 | 9.404 |
| shuffled | -11.874 | -12.698 | -12.543 | -12.413 |
| stop | 0.815 | 0.706 | 0.815 | 0.706 |

The recurrence improves on flat and shuffled-label students, with 100 more active parameters than the flat arm. No direct recurrent policy wins any of the eight roots per cohort and cap against reuse-first after charging work. At 32,768, the raw recurrent utilities are 14.679 and 16.712, against reuse-first 26.996 and 25.762. Thus the deficit already exists before adding planner cost. At 8,192 frequent fallbacks further hurt performance. Both calibration gates disable the scorer because no threshold yields a positive sampled-state priced-regret gain. Gated ties reproduce reuse-first without neural scoring; they are not learned-policy improvements.

No promotion. Selected oracle/myopic states provide limited training coverage, and a predicted continuation-value policy can enter new states. These are plausible mechanisms for the deficit, not established causal explanations. Next: collect exact on-policy counterfactual labels after recurrent deviations, separately sample states where cheap reuse-first has positive regret, predict residual improvement against that policy, and gate before expensive feature acquisition. Compare cheap rule, residual fit, and equal-work shuffled controls on fresh root-disjoint public development cohorts.

## Native COE certificate bridge

The pinned native observer solver and independently checked code/clique certificates establish exact ordinary minimum widths for the planted graph family. On 24 roots, cold solves = 96; reuse solves = 48; 48 successful transports; 24 invalidated transports rejected and replaced by fresh cold solves. All 96 compared widths agree. CPU for the complete cold loop was 18.703 ms and for reuse 17.607 ms, including verification and rejection. The 50% native-call reduction is exact for this roster; the single small timing comparison is descriptive and insufficient to establish a general speedup. The verifier deliberately rejects graphs where its clique bound cannot establish the exact width (e.g. odd cycles).

The student and the COE bridge are separate experiments. No learned native certificate policy, Lean kernel replay, native DataGoblin model update, provider call or training-data admission occurred. No GPU or API spend.
