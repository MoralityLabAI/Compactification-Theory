# SHAFT-COMP-24: Learning a temporal transition rule from behavior

**Preregistration:** 2026-11-11, before COMP-24 calibration, rule discovery or held-out evaluation. Continuation of SHAFT-COMP-23 and PR #96. These are frozen small synthetic GRUs with intact weights. The experiment does not involve external devices, covert persistence, or checkpoint recovery.

## Hypothesis

COMP-23 demonstrated a 900-parameter behavioral-only factorized decoder when it was supplied the *correct* six-by-six Boolean trigger function T(p,a) for a persistent latch. COMP-24 asks whether T itself can be recovered using only observed prefix tokens and native-model one- and two-query behavioral responses. No oracle latch, goal, trigger, or internal-state labels are available to the rule learner.

For an effective goal table b in {0,1}^6, a previous observed input p and two new queries (a,c), the structural model predicts

  B1(c) = b[c],
  B2(a,c) = b[c] XOR T(p,a).

The learner is told only this XOR **factorization family**, not whether T is the repeat-toggle or successor-toggle rule. It receives the previous input p as the last token of the *observed input history*; in this finite task p is available externally. The rule is a 6x6 binary matrix. The Boolean evidence for its entry is B1(c) XOR B2(a,c), separately for each c. Pool those 6 observations over independent calibration contexts with last input p; predict T_hat[p,a]=1 iff the pooled count of ones is strictly greater than half (ties predict 0). No posthoc threshold search or smoothing.

The exact finite-state reference must establish that B1 and B2 uniquely determine T when previous-input p is observed, prove this by XOR cancellation, and give an explicit example that an unlabelled row permutation remains observationally indistinguishable if p is hidden. The finite law is proved separately from trained-model evidence.

## Model cohort and calibration/heldout separation

Use the *previously trained, frozen* SHAFT-COMP-23 checkpoints from its immutable receipts; **no base-model retraining**. Original model seeds are already known from COMP-23, so this study is a fresh-trajectory and protocol holdout, not a new-model holdout.

Calibration sources (only used to learn T):
- repeat_toggle model seed 2301.
- successor_toggle model seed 2311.

Heldout model seeds (never used to fit or revise T):
- repeat_toggle seeds 2302 and 2303.
- successor_toggle seeds 2312 and 2313.

All models use 24-state-dimensional GRUs, six query symbols and 64 synthetic goal embeddings. Check each immutable checkpoint's SHA-256 and the expected model state_dict digest against the COMP-23 original receipt. Report previous 12/32-step native accuracy and refuse mismatched checkpoints.

## Calibration and freeze before holdout

For each family, sample 64 goals ×16 independent input prefixes at each recurrent depth 12 and 40 (2,048 calibration states per family), using seeds (24000000 + source_seed*101 + depth*131) for data and (source_seed*100000 + depth*17 + 24) for random input prefixes. For each frozen state read its last *observed input token* p. Collect all 6 native one-query output bits B1 and 36 native two-query final bits B2. Pool binary evidence by p,a across all six c positions. Freeze all 36 inferred trigger bits, counts of evidence votes, per-entry vote margins, source model hashes, calibration full receipt SHA and an identification status in a **separate Git commit** before running any heldout model evaluation.

The primary exact-rule result is number of inferred entries agreeing with the known finite transition rule, measured *after* the inferred matrix is frozen. Report full matrix for both families and row coverage. A learned rule can be wrong; do not alter training examples or majority rule after observing this fact.

## Heldout test, frozen metric

For each of the four heldout models, draw *fresh* 64 goals ×16 input prefixes at depths 12 and 40, with seeds (25000000 + seed*101 + depth*131) for data and (seed*100000 + depth*17 + 37) for random prefixes, separate from calibration. Compute native B1 and B2 outputs in identical model states. Reconstruct B2_pred(a,c) = B1_native(c) XOR T_hat(p,a) using the model's **current measured behavioral signature** and its observed previous input. Compare against the native frozen model's actual B2, never against goal-derived oracle labels for the primary evaluation.

Controls:
- zero_trigger: B2_pred(a,c)=B1_native(c), corresponding to no state transition.
- wrong_family: trigger matrix from the other family, which uses the same calibration scheme.
- oracle_trigger: the true finite automaton matrix, for *posthoc reference only*.
- permuted_previous: relabel p as (p+1) mod 6 before looking up T_hat, which tests whether the observed input context is material.

Primary metric is *pooled heldout native two-query bit accuracy* of learned T_hat versus zero_trigger, at depth 40, with micro-accuracy and exact 36-bit signature accuracy also reported for all four methods at depths 12 and 40. A paired, seed-stratified 1,000-bootstrap 95% interval for the depth-40 micro accuracy improvement is registered. The support criteria are (a) all 36 learned trigger entries exact in **both** families, (b) positive 95% bootstrap lower bound for improvement versus zero_trigger, and (c) improvements in all four heldout model seeds. Report any failure verbatim.

Secondary: evaluate whether COMP-23's 900-parameter factorized architecture can use T_hat in place of known T. This requires separately retraining the behavioral-only factorized head on the same native B2 targets with T_hat fixed, and comparing to the known-rule control under identical initialization and 600-step schedule; this secondary may be omitted if it exceeds CPU budget. Do not infer a new model-level holdout from it.

## Limits and reproducibility

The learner knows the XOR compositional form and has access to observed input history and counterfactual native behavior. It does not discover an arbitrary unknown program, and its output depends on the full frozen model being available to query. Task IDs are used only to balance evaluation states and for posthoc accuracy checks, not to infer T. All training/evaluation is local. Reproduce frozen results with PyTorch 2.10.0+cpu, NumPy 2.3.5 and 3 deterministic CPU threads; pin original checkpoint hashes and exact JSON replays. Standard-library finite tests, the XOR Lean theorem and the Python receipt audit run in lightweight CI.
