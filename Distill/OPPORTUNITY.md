# DISTILL-09 / Certified research opportunity detection

DISTILL-08 retained the hand-written budget rule as the strongest tested
reference policy. This independent opportunity track tests whether a small,
exact, metered calculation can justify departing from that rule. It does
not modify the external-teacher qualification track.

## Certificate contract

Let pi denote the budget rule and P=4096. All continuation values refer to
the public finite belief state. Measure them in integer units of compatible
world mass times priced utility. For a first action a, compute lower and
upper bounds on taking a and then using pi. Future incumbent selection work
is included in both continuations; the current incumbent selection was paid
and is common.

For compatible mass M and charged optional work W (including prior and
current gate checks), accept only when:

    lower(a) - upper(pi(s)) - M * W > 0

The numerator is in mass-times-priced units (raw reward multiplied by P).
Experimental credits are deducted in the environment reward. Optional search
work is a distinct deduction. A receipt binds the public state, source spec,
incumbent, completed interval trials, selected action, cost, remaining work
reserve and positive margin. An evaluator-only exact DP and independent
per-world replay check the accepted signed advantage. The actor never sees
these oracle values.

At an unexpanded tail with b experiment credits and r projects remaining,
the incumbent's expected raw reward is at least -b: it uses nonnegative
conditional expected terminal decisions, with total experiment cost at most b.
The raw upper bound is r times the certificate reward. Future incumbent
selection work is at most 64*b + 468*r under the registered logical
accounting. A nonterminal action costs at least one experiment credit;
the meter charges at most 64 units for its incumbent selection. A terminal
selection costs at most 468 units and advances one project. Search admission
preserves this conservative continuation reserve. These bounds are checked
over reachable small states and expanded rollout nodes.

The general bounded tree propagates mass-weighted intervals, with a local
memoization table charged for lookup. At the last project, when a direct
audit is unaffordable and the incumbent would stop, a specialized exact
one-experiment calculation computes observation partitions and the best
terminal recommendation or defer for each. Its future incumbent terminal
selection costs 379 work units under the existing budget rule. The same
exact verifier checks this shortcut.

A trajectory attempts at most one search. After accepting or rejecting,
it returns to the budget rule, so accepted margins describe its actual
remaining continuation. Failed, censored and unprofitable searches still
cost work. Positive conditional margins do not guarantee whole-policy
improvement: other histories may pay for rejected searches. Forced-incumbent
controls run the identical search while declining the selected action.
Raw parity with the budget rule is required.

## Frozen campaign

Calibrate on eight ordinary roots (seed 2026112851), then evaluate 12 ordinary
(2026112852), 12 shifted-control (2026112853), and 12 scarcity-shift roots
(2026112854), with distinct public spec hashes. Scarcity modifies initial
credits to 4–7, tilts the law prior toward a negative premise, and varies
control cost. No scarcity tuning.

Each root replays all 64 worlds under the exact prior. Compare budget rule,
late gate (2048 search units; exact terminal shortcut), bounded tail gate
(4096 units, depth 5), broad gate (8192 units, depth 9), their three
forced-incumbent controls, and a calibrated policy. Calibration selects one
configuration per cap, requiring positive mean priced gain, wins on at least
half the roots, and no rise in mean false recommendations. Otherwise disable
optional search.

Caps are 8192 and 32768 per trajectory; price is unchanged from
DISTILL-07/08. Report raw utility, priced utility, gate costs, completed
trials, accepted certificates, local margins, paired wins, work and
actor-only process CPU. Logical work is an abstraction, not strict matched
CPU time. Evaluator receipt auditing, oracle checks, and environment
transitions are excluded from actor timing and work for every arm.

    python -m Distill.opportunity_campaign --out runs/research-opportunity-v1

The hosted workflow archives full synthetic root receipts and completed
search certificates. No learned fit, model-generated answer, API request,
GPU training, native DataGoblin update, or downstream training admission.
