# DISTILL-10 / Counterfactual cancellation and risk-aware research opportunity certificates

The DISTILL-09 result established locally profitable deviations but found
negative mean priced utility once unsuccessful search was charged. This
revision preserves the budget rule and evaluates an alternative bounded
certificate engine. No previous root results or native gym are changed.

## Signed frontier

Let T(s) be the incumbent continuation in unnormalized compatible-world
mass times priced utility; let R(s) be the mass-weighted number of false
recommendations. The actor expands the signed difference between taking a
candidate action and taking the incumbent action. Every term is keyed by
the complete public sufficient state, including the belief mask,
remaining experimental credits, and evidence/certificate flags. A common
state with equal opposing multiplicity cancels both T and R exactly.
Processing maximal rank (remaining stages and credits) before descendants
allows reconvergence to be detected before a common state is expanded.
Distinct belief masks or budgets are never assumed equivalent.

For a frontier state of mass M, b remaining experimental credits and r
remaining projects, the utility interval is
[-(4096*b+reserve)*M, 4096*r*certificate_reward*M].
The incumbent selects terminal decisions with nonnegative conditional
expected raw reward, so the false-recommendation interval is
[0, floor(r*M*correct_reward/(correct_reward-wrong_reward))].
Sign reversal swaps lower and upper endpoints. Exact immediate rewards,
risk masses, selection work, and successor intervals are combined in the
signed sum. A partially exhausted expansion leaves its previous valid
interval intact. Metered logical work includes mass scans, action
evaluation, branch construction, frontier scans and bookkeeping.

Admit a single deviation only if the lower utility difference exceeds
the mass-weighted cost of all gate checks and optional search, and the
upper risk difference is <= 0. The actual continuation then returns to
the same budget rule with sufficient reserved work. This certifies
conditional expected incremental risk under the stipulated finite model,
not individual-world correctness or real-world scientific safety.
Gate failures still cost work. A separate evaluator computes exact
continuation values and explicitly replays accepted actions over all
compatible worlds.

The split ablation uses the same frontier engine and metering but keeps
the two signs in different state keys. Both arms use the same
last-project exact shortcut and cheap candidate screening. This makes
the cancellation contrast identifiable under the declared work model.
Terminal-to-terminal comparisons can cancel completely but are skipped
in actual search when the incumbent terminal reward dominates. The
targeted test measures that algebraic case separately, rather than
claiming an artificial cancellation win in deployment.

## Frozen benchmark

Eight calibration roots and three twelve-root evaluation cohorts:
ordinary, shifted-control, and scarcity shift. Seeds 2026112861 through
2026112864. Each root replays all 64 worlds with its exact prior. Work
caps 8192 and 32768, price 1/4096 utility per unit, maximum one
intervention. Search configs late (2048), tail (4096), broad (8192);
each has paired, split and forced-incumbent arms. Compare with original
DISTILL-09 late and bounded search, which do not certify risk. Calibrate
among paired configs only using complete calibration trajectories;
positive mean priced gain, wins on at least half of calibration roots,
and no increase in root-conditional expected false recommendations are
required. Otherwise disable search. No evaluation tuning.

Report risk deltas, paired/split work, expansions, cancellations,
conditional certificates, all-world priced utility, root wins and
process CPU time. Raw parity is required for forced controls; risk
nonincrease is required at every root for certified actors. Full
per-root and certificate receipts are archived by the hosted workflow.

    python -m Distill.cancellation_campaign --out runs/research-cancellation-v1

The separate-frontier comparison is logical-work matched, not strict CPU
time. Synthetic reference policy only; no provider call, native DataGoblin
weight change, or downstream training admission.
