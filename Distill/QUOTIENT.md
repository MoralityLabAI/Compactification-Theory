# DISTILL-11: research-state compactification

DISTILL-10 found no literal-state reconvergence in its bounded research
search. This independent follow-up asks whether a smaller, policy-relative
state makes counterfactual cancellation possible. The previous campaign
and production code remain unchanged.

## Quotient

Fix a single Model and the frozen budget rule pi. A quotient is valid only
for the continuation under pi. Arbitrary experiment choices can observe
variables omitted by this quotient.

For stage j, keep the stage, the certified-claim flag, and the law and anchor
audit flags when j < 3. The budget rule never selects screen or control.
Consequently their history flags do not affect its choices or selection work.
At the last stage, the law and anchor flags are irrelevant to all remaining
incumbent choices.

Let D and T be direct and transport credit costs, L the law-audit cost, and
r the number of uncompleted projects. For j < 3, saturate credits at
(r - 1[certified])*max(D,T) + (L+T)*1[j<=1 and law unknown].
The threshold covers all remaining certification experiments and the
law-audit-plus-transport affordability guard. If already saturated, every
selected experiment decreases the threshold by at least its cost; advancing a
project preserves saturation of its successor. Unsaturated budgets are kept
exactly. For j=3, all budgets with direct audit unaffordable are equivalent;
all with direct audit affordable are equivalent; certified states have no
remaining experiment budget requirement. The quotient preserves exact chosen
actions and declared selection work, including the certificate override
and the budget-rule guard.

The flags arm keeps the original mask. The marginal arm projects the
compatible-world masses onto future truths, including the law bit through
stage 1 (the last stage at which pi can select a law audit). Divide the
integer vector by its gcd g. The canonical vector determines conditional
expected reward and false-recommendation risk. For a selected experiment,
conditioning on a projected bit and dropping obsolete coordinates preserve
the quotient successor distribution and integer divisibility by g. Transport
has a deterministic checked successor. Terminal choices drop the completed
truth. The same normalized distributions determine the same terminal argmax
and tie rule. Therefore induction over remaining stages and credits yields
equal conditional utility (with future planner work) and expected risk within
a class. Unnormalized values need not be equal: a representative with mass
factor g carries g times the canonical value. The frontier sums signed
integer mass factors; scaling through a representative is exact and division
is checked. Class rank strictly decreases along the incumbent transition.

This proof argument assumes the registered finite model, legal reachable
beliefs, and the exact budget rule. Local congruence is audited by independently
computing the selected action, metered work, normalized immediate reward and
risk, and weighted successor class distribution for every state in complete
candidate continuation closures. Tests also enumerate all legal reachable
states in two small ordinary/shifted fixtures. No claim about arbitrary
research policies, hidden states, or a Lean kernel proof follows.

## Diagnostic and bounded campaign

The diagnostic starts from every state reachable under pi, uses the registered
broad gate and candidate filter, and enumerates both complete incumbent
continuations following the first action. It counts candidate pairs with a
shared nonterminal class in each key mode, and pairs with quotient overlap
but no literal overlap. These set intersections are potential reconvergences,
not measured cancellation: masses and bounded expansion order still matter.
All states in those closures are audited for local congruence. Terminal states
are excluded from class counts to avoid trivial compression.

The actor uses the same signed interval and risk acceptance rule as DISTILL-10.
Literal, flags and marginal use the same new frontier and candidate order,
with quotient construction explicitly charged. The previous paired engine is
also replayed. All modes share the previous exact late shortcut. The marginal
meter charges world-slot scans and bin/GCD reduction before forming the key;
this is a declared work metric, not a CPU-time cap. A partially exhausted
expansion retains its valid prior interval. The evaluator checks every
completed interval against exact continuation values and independently replays
accepted actions. All forced controls must preserve raw incumbent aggregates;
no certified root may increase expected false recommendations.

Frozen roots: 8 calibration and 36 evaluation (12 ordinary, 12 shifted,
12 scarce), seeds 2026112871 through 2026112874. All 64 worlds per root.
Work caps 8192 and 32768. Arms: budget rule, previous paired tail/broad, common
late shortcut, literal/flags/marginal tail and broad, forced flags/marginal
tail, and calibration-selected policy. Calibration selects a positive-priced,
non-risk-increasing policy with wins on at least half the calibration roots,
or disables search. No evaluation tuning. Compare the new modes to the new
literal arm, not to the prior engine for an isolated key-cost claim.

    python -m Distill.quotient_campaign --out runs/research-quotient-v1

The hosted workflow archives full synthetic rows, diagnostics and completed
certificates. No paid model, provider answer, GPU training, or native
DataGoblin parameter update. Downstream training admission is false.
