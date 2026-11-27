# DISTILL-08 / Learning profitable research deviations

DISTILL-07 found that a direct recurrent policy lost to the reuse-first rule on
every tested root. This experiment retains reuse-first as the incumbent. It
does not change the longitudinal environment or any previous result.

## Target

For each reachable public sufficient state, compute two independent labels:

- optimal-continuation residual: Q*(s,a) - Q*(s,reuse-first(s));
- incumbent-continuation advantage: Q^reuse(s,a) - V^reuse(s).

The second label measures the value of a single deviation with incumbent
continuation, rather than giving a learner credit for a hypothetical optimal
follow-up it cannot execute. Both labels are exact rational values over the
finite conditional prior. The first action is counterfactually varied and the
remaining policy is replayed. These labels are evaluator-only. A positive
single-deviation value does not guarantee benefit from composing multiple
learned deviations, so calibration and evaluation replay entire trajectories.

Collect all reachable sufficient states under incumbent and exact-oracle
policies on 24 fresh training roots. Fit a provisional ridge residual ranker,
collect additional states reached by it on the same roots, then refit on the
union, with up to 96 retained states per root and priority for novel states and
positive one-deviation opportunities. Novel states are a coverage diagnostic;
they need not all be mistakes. The provisional collection has no trajectory
work cap; final evaluation does. Use synthetic labels only. Train a
within-state shuffled-label control on the same rows and bounded weighting.

The scorer is a deterministic CPU ridge model with one head per action, using
the prior public-belief features, incumbent-action indicator, and bias. It
predicts incumbent-continuation advantage. A cheap gate keys on stage,
incumbent action, remaining-credit bucket and certificate availability.
Train-only gate statistics use retained incumbent states, root-weighted means,
at least three contributing roots and six states. Gate lookup precedes dense
feature acquisition. A work-reserve check protects future cheap choices.
Neither hidden-world information nor oracle values enter actor features.

## Registration

24 train, 8 calibration, 8 ordinary evaluation, 8 shifted evaluation roots,
fixed new namespace seeds 2026112841–2026112844. No shifted training or
calibration. The two logical work caps are 8192 and 32768 per world, with
the same declared cost price as DISTILL-07 (one utility unit per 4096 units).
Host-supplied public sufficient-state construction and referee costs excluded
for every arm. CPU timing around actor selection is measured separately,
including optional scoring, with a descriptive 25 ms envelope.

Compare reuse-first, a hand-written budget exception rule, always-score
residual, fixed-gate residual, fixed-gate shuffled labels, fixed-gate forced
incumbent (overhead control), and calibrated residual. The fixed gate and
margin both equal .25 utility units. The calibrated gate/margin grid is
{.25,.75,1.5} squared. Choose separately by cap, on calibration trajectories
only: positive mean priced utility improvement, wins on at least half the
roots, and no increase in average false recommendations; otherwise disable
scoring. Freeze before evaluation. Evaluate all 64 worlds per evaluation
root. Report root-paired priced utility deltas, raw utility, metered work,
deviations, score calls, fallbacks, certificates and false recommendations.
Do not claim that a local regret fit is an improved whole-program policy.
The forced-incumbent arm charges the actual scoring path and reports any
budget-induced outcome changes, rather than assuming raw parity.

Run:

    python -m Distill.residual_campaign --out runs/research-residual-v1

Fitted weights and complete root receipts remain local. No live provider
calls or native DataGoblin checkpoint training. This is an isolated reference
student with synthetic supervision and training_admission=false for downstream
ingestion. Logical work is a declared accounting model, not strictly equal
CPU time. Results remain public development; no automatic promotion.
