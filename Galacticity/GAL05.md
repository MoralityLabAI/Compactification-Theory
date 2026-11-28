# GAL-05: Computational capital theory
Status: exact finite planning model, mathematical proofs, synthetic portfolio
campaign, and core-Lean arithmetic candidates. No real-money, native BPD,
or DataGoblin training claims.

## 1. Capital contract and path algebra

Capital is an explicitly declared budget of fungible work units. At each
one-shot acquisition i, pay c_i>=0 before realizing r_i>=0 in verified
savings before the next acquisition. No borrowing or free advance on
future savings. Returns must be nonoverlapping relative to a fixed
baseline, and costs include whatever verification is required by the
resource contract. A realized saving is a budget credit; an accounting
estimate cannot be used as if already earned. Idle capital is permitted.

For an ordered sequence, the prefix balance before i is
K + sum_{j<i}(r_j-c_j). Thus the least feasible K is
max(0,max_i(c_i-sum_{j<i}(r_j-c_j))). This follows by intersecting all
funding inequalities and is attained by replay. The net gain G is
sum_i(r_i-c_i). The pair (N,G) summarizes feasibility and terminal
capital: given K>=0, feasible iff K>=N, and terminal balance K+G.
For any valid summary N>=0 and N+G>=0.

Sequential composition: (N1,G1)*(N2,G2) =
(max(N1,N2-G1),G1+G2). The identity is (0,0). Associativity follows
by expanding the required capital into max(N1,N2-G1,N3-G1-G2)
for either parenthesization; the gains add. Closure uses the fact
that executing two feasible paths leaves nonnegative capital.
This is a resource-summary monoid; the equivalence is contextual only
when future actions depend on capital and the externally visible state
has also been preserved. More generally at the same terminal state
and elapsed time, N lower and G higher gives a dominating path label.
Both prefixing and suffixing a fixed summary preserve this dominance.
A path-frontier method can retain undominated labels when optimizing
initial capital and terminal gain jointly. If time or memory differs,
those coordinates must be carried in the label.

## 2. Independent ordering theorem

When every item is required exactly once, returns do not depend on
order, all prerequisites are absent, and there are no deadlines or
holding charges, order nonnegative-net-gain projects by ascending c,
then negative-net-gain projects by descending r.

Adjacent exchange proof: the two-item initial requirement for A then
B is max(c_A,c_A+c_B-r_A). For two profitable projects with c_A<=c_B,
this is <=c_B, while reverse requires >=c_B. For two loss projects
with r_A>=r_B, each term in the first maximum is <=c_A+c_B-r_B
(which is a term of the reverse maximum). For profitable A and
loss-making B, the first maximum is <=max(c_A,c_B), while the reverse
maximum is >=both costs. Each exchange preserves final gain, so cannot
make later suffix feasibility worse. Repeated exchanges produce the
stated optimum. Ties arbitrary. Sorting costs are outside the theorem.

Example: d sequential projects cost 2^i and return 2^(i+1), i=0..d-1.
Starting at K=1 yields exactly 2^d after d investments. A final target
costing 2^d is reachable; its own return is immaterial to acquisition.
Without the smaller investments, initial capital 2^d is necessary.
This is an explicit model family, not an empirical scaling claim.
Realized independent savings must exist; returns are not created by
moving money between accounts.

Replay gal05_paths.mjs: 27,782 assertions; 6,561 pair order controls,
4,096 associativity/three-item permutation cases, 96 six-item exhaustive
permutation cases, and exact doubling examples through depth 40.

## 3. Finite-horizon recurring portfolio

State (t,M,K): elapsed ticks t, active project bitset M, cash K.
Known nonnegative project cost c_i, per-tick yield y_i, payload s_i,
prerequisite mask P_i. An acquisition is legal only if prerequisites
are active, project not active and total payload remains <=cap.
At most one project may be acquired per tick; idle is always a legal
structural choice. Active projects persist for the entire episode,
with no eviction, invalidation, holding charge or overlap in savings.
Each tick pays a common fixed action fee f and the selected construction
cost upfront. If affordable, new portfolio M' earns
Y(M')=sum_{i in M'}y_i at the end of the tick. The new capital is
K' = K - f - c_i + Y(M'), with c_idle=0. Success is acquisition
of a specified target by deadline H; no requirement on terminal
profit. Target acquisition is terminal after the current tick.
This is a fully known finite planning environment, not a hidden-future
online market. Costs are synthetic work-unit budgets.

Exact reverse Bellman recursion for required capital:
F_H(M)=0 if goal in M, infinity otherwise.
For t<H, F_t(M)=0 if goal in M; otherwise
F_t(M)=min_{legal actions i} [f+c_i+max(0,F_{t+1}(M')-Y(M'))].
The term is the least K satisfying both immediate affordability and
the next-state funding constraint. Backward induction proves exactness,
including impossible states. The algorithm uses O(H*2^p*p) candidate
evaluations after mask-income/size preprocessing; p is project count.
Keep action choices to recover a feasible witness. Value-only memory
O(2^p), or O(H*2^p) with full witness actions. This exponential planner
is an oracle, not a free deployable compactification.
Independent forward recursion stores maximum cash at each reachable
(t,M): higher cash with identical time and mask dominates lower
cash. Every action is cash-monotone. Comparing F_0 against forward
reachability at K=F_0 and K=F_0-1 verifies the boundary. Tiny independent
brute recursion checks all action paths for each tested seed.
Additional cap obstruction: with persistent projects, union of active
and all mandatory ancestor/target projects must fit cap. Reserving
space for this closure prevents a class of greedy dead ends; it does
not prove deadline or cash feasibility.

## 4. Counterexample to greedy ROI

Cap 4, H=6, zero fee, seed 2. Project A costs 2, returns 8/tick,
uses all 4 payload units. B costs 2, returns 2/tick, uses 1 unit.
Goal costs 9, requires B, uses 2 units. Greedy income and ROI choose A
and permanently block the goal, despite accumulating capital.
Exact minimum seed is 2: acquire B, idle four times, acquire goal.
The reserve-aware policy succeeds. This obstruction depends on the
persistent/no-eviction contract. Eviction would change the state model
and could remove this particular dead end.

## 5. Synthetic benchmark

Run node Galacticity/gal05_portfolio.mjs. All projects and resource
contracts are public; the DP minimum and optimal witness are grader
information. The policy is given the requested initial budget, which
is constructed from the oracle minimum or minimum+10 for difficulty
control; thus evaluation is conditioned on feasibility, not on a
random deployment budget distribution. Parameters and future income
are known. No model calls, learned weights, or native savings involved.
Actions are replayed with feasibility checks. The declared common fee
(0 or 1) is charged on every tick in all arms. Policy scan/score counts
are recorded separately; actual DP/planning time and all computation
beyond this fixed fee are not priced. Consequently success comparisons
are planning quality, not end-to-end speedups.

Calibration: five roots each in four families (independent, ladder,
dependencies, distractors), two capacities (6,9), two fees, two budgets,
160 cases. Choose one of six fixed heuristics by success count, then
successful ticks, then scan count. Selected reserve_value. The heldout
grid uses eight disjoint roots per family plus a new short-deadline
family, two capacities, two fees and two budgets: 320 cases from
160 environment configurations (40 root seeds; related cases paired).
All generated main-grid cases feasible at some seed; tiny controls
include impossible environments. No independent external test set.
The minimum-seed boundary is checked for each feasible environment.
Tiny brute: 96 small environments x 17 seeds = 1,632 cases, each
checked against forward and backward planning. Total 6,937 checks.

Heldout success /320 (at exact oracle seed /160):
wait 0 (0); greedy income 111 (53); ratio 124 (58);
ancestors-only 154 (63); reserve 244 (95); calibrated reserve_value
241 (95). Oracle succeeds on 320 by construction. Calibration winner
loses to the alternative reserve heuristic by three test cases;
retain the original selected policy and report the negative comparison.
New short family: selected 57/64; distractors 27/64. This is not a
trained DataGoblin result.

## 6. DataGoblin research gym interface

Expose public project descriptions, time, active mask, capital,
fee/cap and target. Ask for an action sequence, per-tick cost and
savings ledger, claimed minimum seed, and an explanation of constraints.
Grade witness legality with replay; grade minimum against the exact
reverse oracle and separately check impossibility via forward search.
A feasible plan is distinct from a proof of optimal seed. Public
packets must omit oracle and optimal plans; hold out entire project
families and prevent test tuning. The checked-in receipt is a public
development/evaluation fixture containing solutions, unsuitable for a
blind training split. Add measured controller costs before making
economic claims. No actual training admission or DataGoblin changes.
For TRM, reinvest savings only after a matched task-quality verifier,
and use source/version epochs for any resource-dependent gains.
For BPD and DISTILL, do not add savings from overlapping computations;
same-workload independent accounting must be established first.

Next: stochastic returns with delayed verification, correlated payoff
and loss, competing project frontiers, eviction and expiring budgets.
Need distinguish capital feasibility, regret, and cost of the planner.
