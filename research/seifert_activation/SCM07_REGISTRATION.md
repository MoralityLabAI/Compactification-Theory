# SCM-07: source-local Jacobian registration, 2026-11-13

**Research status:** Source-level mathematical extension and CPU mock study, registered before trained-model runs. It continues the draft SCM-01 through SCM-06 stack without changing RMP's gate G0 or sealed results.

## Prior and test order

1. Verify the pinned four RMP Python blobs using `scm06_audit.verify_rmp_source`, then confirm checkpoint and D-shared hashes from SCM-04 captures.
2. Run the **source-global tokenwise** bound on SCM-02's 64 selection-half group-disjoint calibration tasks, four families balanced, before inspecting captured braids. This is the primary new analytic method and requires no assumed clean centers.
3. For the second, explicitly conditional analysis, capture clean pre-LN1 and pre-LN2 middle states with read-only hooks at each expanded visit. These are diagnostic floats until independently enclosed. Derive externally verified per-token error margins before labeling a local bound independently certified.
4. For each sample/pair/visit compare the source-global and local-LN pair bounds against SCM-05's exact rational clearance thresholds. Use `N=32` and `N=64` samples with `2π/N<44/(7N)`. Respect SCM-04's token-identity ordering and closed-braid rejection rules.
5. Report expected collisions, null disks and model outcome/regret changes with source-preserving controls; keep the causal test separate from the topology observation.

## Fixed model protocol

- `LOOP-T s1` and `FF-U s3`, eight expanded visits, input reinjection for FF-U according to the authoritative RMP `AnalysisModel` registry.
- Injection after visit zero at token four. Interventions use the existing frozen two SAE decoder rows and radius `0.05` times the clean token RMS, with tokens 16, 32, 48 observed.
- A strict causal negative control injects after visit zero at token 63 and observes tokens 16, 32, 48; the ideal model must have zero angular derivative at those locations for every visit.
- No new trained checkpoints, teacher traces or unblinded held-out data may be used to tune the mathematics. Retain all refused/collision cases.

## Proof-status states

- `source_global_unconditional`: valid in exact real arithmetic for the pinned source graph and exact stored weights, without additional clean-center hypotheses. This means unconditional on data-dependent *centers*, not unconditional on architecture identity.
- `conditional_on_external_ideal_center_enclosures`: sound only if an independent verifier has bounded the ideal clean states relative to the supplied center floats at *every* listed site and token.
- The offline `scm07_audit.py` JSONL reader treats all imported derivative bounds as claims. It does not inspect checkpoint weights or source hashes; source-locked recomputation and provenance checking are separate requirements.
- `observed_link_certified=false`: required until real output and trigonometric/forward numerical error envelopes have been verified; a numerical sample agreement at 32 and 64 angles is insufficient.

## Falsification gates

- A local-bound pass without a valid center enclosure is reported as conditional and excluded from primary success.
- If the prefix zero control changes under ideal causal evaluation, review source pin and token indices before considering the topological result.
- A source-global bound that stays above all sampled clearance thresholds constitutes an inconclusive outcome even if the observed braid words are stable.
- A topology change that fails to predict causal outcome beyond random-chart and shuffled controls is null evidence for the proposed reasoning-topology application.

### Separate remaining engineering requirements

For actual checkpoint deployment, preserve the existing no-new-dependencies policy on the worker, do not interrupt GPU tasks, keep baseline outputs immutable, and use distinct audit receipts. An independently sound numerical center-enclosure algorithm and activation-output error analysis are still needed.
