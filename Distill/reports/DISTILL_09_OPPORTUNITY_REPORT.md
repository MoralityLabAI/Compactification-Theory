# DISTILL-09 / Certified research opportunity detection

Tested commit: `3e3301dc5c7a7d193811b28ade7a355f32f1d525`. [Opportunity workflow](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46037850500) completed successfully (five targeted tests and full campaign). Full synthetic root receipts and completed certificates were uploaded as [distill-09-full-receipts](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46037850500/artifacts/15219606247), with 30-day retention. Report ID: `fab571d8b8975d4ec1fd397d6b39db3a00ee89a2b8d34a23a93d777f9b2868de`. The printed aggregate is archived beside this report.

## Frozen results

Eight calibration roots and 36 distinct evaluation roots (12 ordinary, 12 shifted-control, 12 scarcity-shift), 64 worlds per root. Two logical work caps, eight arms per cap. 274,560 evaluation world transitions; 2,184 audited intervals across arm/root/cap receipts, 48 independent priced action replays, zero CPU envelope overruns. These counts include repeated controls and caps. The finite model is synthetic and the work metric is not strict CPU time.

| Policy | Ordinary 8192 | Shifted 8192 | Scarce 8192 | Ordinary 32768 | Shifted 32768 | Scarce 32768 |
|---|---:|---:|---:|---:|---:|---:|
| budget_rule | 24.6988 | 25.3317 | 10.0305 | 24.6988 | 25.3317 | 10.0305 |
| late | 24.5939 | 25.2392 | 9.8251 | 24.5939 | 25.2392 | 9.8251 |
| forced_late | 24.5793 | 25.2392 | 9.8008 | 24.5793 | 25.2392 | 9.8008 |
| bounded | 24.2628 | 25.0299 | 9.1660 | 24.2628 | 25.0299 | 9.1660 |
| forced_bounded | 24.2628 | 25.0299 | 9.1660 | 24.2628 | 25.0299 | 9.1660 |
| broad | 23.1996 | 23.8208 | 8.5163 | 22.6920 | 23.3238 | 8.1242 |
| forced_broad | 23.1996 | 23.8208 | 8.5163 | 22.6920 | 23.3238 | 8.1242 |
| calibrated | 24.6988 | 25.3317 | 10.0305 | 24.6988 | 25.3317 | 10.0305 |

## Findings

- The late gate's positive local certificates cover 32 of 768 ordinary and 32 of 768 scarce world trajectories per cap (unweighted counts), with prior-weighted acceptance rates 0.035714 and 0.059524. No shifted acceptance. Minimum accepted net lower margins are 0.14160 and 0.14551 priced utility. Independent replay verifies every accepted unique certificate within each root/arm/cap. These are conditional improvements against the budget-rule continuation; they do not include costs borne by other histories.
- Late gate mean priced deltas are -0.10490 ordinary, -0.09244 shifted, -0.20548 scarce, at either cap. It wins one ordinary and one scarce root per cap, with 34/36 losses. It slightly raises raw utility on ordinary and scarce cohorts but also raises expected false recommendations. Forced-late isolates overhead while preserving incumbent raw aggregates.
- Bounded and broad gates accept no departures. The broad search spends its allowance without completing any candidate trial on ordinary/shifted roots at 8192; even at 32768 these cohorts complete no candidate trials. Search depth and the order of computing the baseline before alternatives are relevant limitations.
- Calibration disables optional search at both caps. Keep the analytic budget rule. There is no learned fit or model promotion here.

Evaluation actor-only process CPU totals 4.462 seconds over all arms and worlds. Logical-work admission reserves the remaining incumbent work under the declared meter. The tests validate the arithmetic lower bound, exact replay, and rejection of a changed certificate margin; they do not establish real-world decision value.

## Follow-up

The next candidate is a paired-difference certificate engine with shared continuation cancellation, cheap impossibility screening and action-specific search allocation. Calibrate a risk-aware gate against the same budget rule before fresh-root evaluation. Add a lower bound on net advantage and a separate upper bound on incremental false-recommendation risk. Preserve this negative result; no retuning on these evaluation roots.

The first combined offline suite ran 64 tests with one error in the concurrent external-teacher test fixture: it attempted to overwrite an existing template using an exclusive-create writer. Concurrent commit a54fee311235862c6d2b86a7bb87dde465b34718 repairs that fixture; this report commit preserves the fix and does not change the production writer or opportunity campaign code. The targeted opportunity workflow was successful before the repair. No API calls, GPU training, native DataGoblin update, or downstream training admission occurred.
