# SCM-05 pre-registered pilot: continuous-braid and quantized-link stability

**Registered:** 2026-11-13, before real RMP capture, before model-derived derivative bounds, and before examining held-out outcomes. Existing sealed RMP results and registrations stay unchanged. SCM-01/02/03/04 synthetic experiments are historical exploratory data.

## Primary question

Does a frozen, externally verified interval derivative bound certify that the observed topology of recurrent activation paths survives finite angular sampling and specified fixed-range low-bit chart quantization? The topology is chart-relative. The causal test of usefulness for reasoning is a distinct downstream hypothesis.

## Fixed experimental contract

- Source cells: registered RMP `LOOP-T s1` and `FF-U s3`, eight visits, frozen checkpoint/artifact hashes. RMP registry and its input-injection settings remain authoritative.
- Dataset: **only** deterministic, group-disjoint selection-half `monitor_calibration` examples, 64 tasks equally divided across the four source families, as chosen in SCM-02. Reporting-half and sealed test cases are excluded from this pilot.
- Intervention: SCM-02's frozen two D-shared SAE decoder directions, fixed injection after visit zero at token position four, radius 0.05 of source token residual RMS; three readout token slots (16, 32, 48). Chart controls include random orthogonal readouts with the original intervention axes unchanged.
- Capture: 32 and 64 angular samples, exact identical endpoint intervention. Record full rejected cases; compare link-level Milnor, crossing, winding and Artin data using the previous instruments.
- New certificate: condition `m > L_pair Δ/2 + 1e-7` for each interval, with 12-decimal outward sqrt. The derivative claim must cover **every angle** of its claimed interval; use measured per-pair relative upper bounds rather than empirically fitted endpoint slopes.
- Quantizer: fixed center-of-bin uniform quantization, no saturation; for the first calibration-only sweep choose the chart range before evaluating that sweep and record it in the immutable bound claim. Try widths `b=1..16` and report the first **worst-case-certified** width, or `uncertified` at the cap. Report the data-dependent width separately.
- Audit: run read-only `scm05_audit.py` over immutable SCM-04 JSONL receipts. Output file must be new, source/claim SHA-256 must be recorded. Any absent/invalid upper bound is a **refusal**.

## Falsifiers and negative controls

- A 4-turn continuous circle sampled at four angles yields apparent zero winding. With correct global bound, sampling must refuse; with 64 angles it must certify four turns.
- A compressed path whose measured pair difference passes through zero must fail the certificate.
- Missing derivative proof and missing bounds yield explicit `missing_external_interval_bounds`; a supplied assumption alone always has `independently_certified=false` in the audit, even if the conditional margin passes.
- Pairwise-exact winding and equal Milnor with failing clearance is **not** accepted as a continuous isotopy certificate. A passing sufficient certificate is only meaningful for the pre-registered chart and model pair.
- Large random chart effects or unacceptably few collision-free trajectories block topology-based causal conclusions.
- Causal relevance: compare any topological changes with prediction/regret changes under feature patches at matched task and compute budgets. A result indistinguishable from shuffled/random chart controls is null evidence for reasoning-specific topology.

## Pending gate

RMP checkpoint bytes, D-shared dictionary and monitor bundle must become accessible to the execution host. An independent verified derivative-bound computation is then necessary. No current result establishes real-model topological compressibility or a Miller-type 4D surface effect.
