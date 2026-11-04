# IDQ-03: goal-visible quotients and recovery-value preservation

**2026-11-04 follow-up to [IDQ-02](MINIMAX_THEOREMS.md).** An exact dynamical quotient can be *vacuously* autonomous: a constant observation map always has one macrostate and zero one-step dynamical defect. A meaningful agent-level abstraction must additionally preserve explicitly declared tasks or goals. The elementary results here formalize that requirement. No claims about consciousness, souls, or biological morphogenesis are made.

## 1. Exact readout and the fibre criterion

Let S be a finite microscopic state set, pi:S->Z a **surjection**, and g:S->{0,1} a goal indicator.

**Proposition IDQ-03/T1.** There exists a macro goal function h:Z->{0,1} satisfying g(s)=h(pi(s)) for all s **iff** the goal is constant on every projection fibre:

    pi(s)=pi(t) => g(s)=g(t).                                (1)

If it exists, h is unique.

**Proof.** The forward implication follows by applying h to equal fibre labels. Conversely, choose any preimage s of each z and define h(z)=g(s); condition (1) makes the choice irrelevant. Surjectivity gives a preimage at every z, and therefore uniqueness. QED.

For r separate Boolean goals g_1,...,g_r, assign each state the signature (g_1(s),...,g_r(s)). Any abstraction preserving every goal **exactly** must distinguish different signatures. Thus if there are m distinct signatures, the macrostate alphabet has at least m elements and a fixed-length label needs at least ceil(log2 m) bits. This is only a lower bound: preserving future transitions can require additional distinctions.

## 2. Exact minimax readout error

Allow probabilistic macro readouts q:Z->[0,1] evaluated against the deterministic Boolean g using the uniform worst-case absolute loss:

    E_goal(pi,q)=max_s |g(s)-q(pi(s))|.

**Proposition IDQ-03/T2.** The optimum min_q E_goal is either 0 or 1/2. It is 0 precisely when the goal is fibre-constant; otherwise it is 1/2.

**Proof.** A pure fibre is represented with the exact 0 or 1 value. A mixed fibre contains a 0 and a 1, forcing max(|q|,|1-q|)>=1/2; q=1/2 attains equality. Each fibre's choice is independent. QED.

For deterministic macro readout h:Z->{0,1}, a mixed fibre instead forces worst-case error 1.

**Nontrivial quotient acceptance criterion.** To prevent collapse into an information-free constant abstraction, report at least two separate axes: (a) dynamic defect from IDQ-02 and (b) goal-readout defect. A combined objective max(E_dyn,lambda E_goal) is meaningful only with preregistered task, scale lambda, allowed interfaces and complexity budget.

## 3. Exact goal-recovery times under interventions

Suppose pi is an exact quotient of **each** permitted microscopic action F_a with macro G_a:

    pi F_a=G_a pi.

For any finite action word w, the existing [MOR-01 Lean theorem](https://github.com/patrickdugan/OntologicalProofs/blob/main/Neoplatonic/LTR/MorphospaceSemiconjugacy.lean) gives pi(F_w(s))=G_w(pi(s)). If g=h pi is a fibre-visible goal, then for every prefix v of w,

    g(F_v(s))=h(G_v(pi(s))).

Thus the *first time* a fixed action schedule reaches the goal is identical in the microscopic and macroscopic systems (including failure to reach before the horizon). The same statement holds for feedback controllers observing only the macro history, provided their policy randomness is coupled and the action/recovery interface is the same. If damage interventions are allowed, they must be included among the actions and satisfy the same commuting law.

The Lean companion in OntologicalProofs MOR-03 formalizes the goal-readout existence criterion and action-prefix hit-event equivalence.

## 4. Approximate stochastic recovery

For finite stochastic kernels K_a and macro L_a, suppose each action's projected micro successor law differs by at most epsilon in total variation, uniformly over **every** microscopic state. If goals are exactly observable through pi and the policy sees only macro histories, [IDQ-02/T2](MINIMAX_THEOREMS.md) couples macro histories with disagreement probability at most

    1-(1-epsilon)^H

by H transitions. Therefore their goal-hit probabilities differ by at most that quantity. The condition includes the target readout: a low stochastic transition error alone says nothing about goals hidden inside a fibre.

### Exact three-step witness

For microstates {0,1,2}, let pi(0)=pi(1)=A and pi(2)=B, with goal B. In one action, microstate 0 enters goal 2 with probability 1/4 per step while state 1 enters goal 2 with probability 1/2, and goal 2 is absorbing. Both otherwise stay in their original microstate. The Chebyshev-optimal macro kernel from A uses success probability 3/8, the midpoint of 1/4 and 1/2; its maximum one-step TV error is epsilon=1/8.

The exact probability of hitting by three transitions is

    P(hit | start=0) = 37/64,
    P(hit | start=1) = 7/8,
    P_macro(hit | start=A) = 387/512.

The larger absolute discrepancy is 91/512. The common coupling bound is

    1-(1-1/8)^3 = 169/512,

which holds. The reference code computes these by backward dynamic programming using fractions.Fraction. This illustrates a conservative general-purpose bound; a tighter problem-specific calculation is available from the exact kernels.

## 5. Evidence status and next experiment

- **Written mathematical proofs:** T1, T2, exact pathwise goal transfer, and inherited TV coupling corollary.
- **Exact finite computational checks:** three new goal tests plus the prior IDQ suite, on Python's rational arithmetic.
- **Lean obligation:** MOR-03 before calling the pathwise goal theorem kernel-verified.
- **Open empirical claim:** a learned *nontrivial* quotient can preserve goal recovery under held-out interventions and offer fewer mutable state bits than the original microstate model.
- **Falsifier:** exhibit states in the same purported macrostate that differ in goal labels or that have divergent projected successors under any registered intervention.

This directly connects Levin's functional language about collective goal pursuit to *operational, interface-relative mathematical conditions*. The philosophical question of whether coherent goals are instances of independently existing forms remains separate.

Run: \`python -m unittest discover -s research/ingression_dynamics_v1 -p 'test_goal_recovery.py' -v\`.
