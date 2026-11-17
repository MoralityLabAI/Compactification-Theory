# ACE-MechInterp Gym v0.1

Standalone CPU synthetic evaluation for task-preserving causal representations and conditional evidence authorization. It reuses the finite-state quotient implementation in research/dynamical_closure/task_closure.py on PR #28. Mathematical ownership of general intervention-preserving abstraction stays with EvalMath. No GPU, model API, external statistical package or signed evidence service.

## Reproduce from repository root

```sh
python -m unittest discover -s research/ace_mechinterp_gym -v
python -O -m unittest discover -s research/ace_mechinterp_gym -q
python -m research.ace_mechinterp_gym.gym --out /tmp/ace-gym
python -m research.ace_mechinterp_gym.verify --out /tmp/ace-gym
```

manifest.json freezes one four-candidate synthetic campaign: root alpha 0.05, four child allocations of 0.01, 128 distinct clusters per candidate, null p0=0.20, alternative q=0.05, one permitted substitution per candidate, plus an explicit flip and erasure on the full-state code. The runner writes trace.json and receipt.json. The verifier checks slot order, same-code pairs, action grammar, synthetic attack schedule and the manifest/trace/source hashes, then recomputes diagnostics. Unsigned hashes establish local replay consistency only.

## Exact finite-state benchmark

State x=(y,b) with y,b bits. Task h(x)=y. Identity action keeps (y,b); reveal maps (y,b) to (b,b). Codes: full state, task bit alone, or constant. One observation samples x uniformly among the four states, samples x' uniformly within its code fiber, then samples an action uniformly from the predeclared action alphabet. This entire pair/action is ONE evidence cluster.

| Candidate | Action set | Smallest exact task quotient | Exact violation probability |
| --- | --- | --- | --- |
| Full (two bits) | id,reveal | 2 bits | 0 |
| Task bit | id | 1 bit | 0 |
| Task bit | id,reveal | 2 bits | 1/4 |
| Constant | id,reveal | 2 bits | 1/2 |

Violation means the post-action task outputs of x,x' differ. The exact task quotient and distinguishing words are structural results. A sampled test only supports a distributional claim for its predeclared pair/action law.

## ACE-style inference

Unsafe null: for each fresh cluster, P(V_i=1 | complete pre-observation history) >= p0=0.2. For fixed alternative 0<q=0.05<p0, use factor e_i=(q/p0)^V_i * ((1-q)/(1-p0))^(1-V_i). Its conditional expectation is <=1 for each null law, yielding an anytime-valid product e-process. For a fixed child alpha, an authorization requires the robust lower-envelope e-value to cross 1/alpha at some prefix. With at most k substitutions of present bits in the *entire prefix* and arbitrary erasures, minimize the log evidence over all compatible clean sequences. Since q<p0, the minimizing sequence sets all missing bits and up to k observed zeroes to one. This O(n) closed form is tested against exhaustive short-trace enumeration.

The real clean trace must remain in that compatibility set at every prefix. Under this condition the lower envelope is <= clean e-process pathwise, so its crossing cannot exceed the clean process's null crossing probability. The child risk bound requires the declared full-history conditional null. A parent risk cap across adaptive candidates additionally requires predictable test/alpha allocation based on the *whole parent history*. The gym pins a single synthetic campaign; it does not enforce live cross-run governance.

The policy-matched robust Bayesian comparator assumes simple Bernoulli(p0) null versus Bernoulli(q) alternative, equal prior odds, and unsafe-certification loss to unnecessary-denial loss ratio 1/alpha. It minimizes the likelihood ratio over the same feasible clean sequences, exactly matching ACE's decisions in this special simple-null setup. No claim of general superiority over calibrated Bayes follows.

## Source dependence counterexample

For task-bit code under id/reveal, one measured pair is a non-violation with probability 3/4. A collector who repeats a single favorable bit 32 times as supposedly independent evidence crosses the 0.01 threshold after 27 copies, leading to false authorization with probability 3/4 under the unsafe null (actual violation rate 1/4). Counting the repeated observation as one source cluster blocks the crossing.

This echoes Beyond Bayes v0.6 source-dependence failures: https://github.com/patrickdugan/beyond-bayes/blob/main/RESEARCH_V06.md

## Scope

The simulator uses a public deterministic pseudorandom seed for reproduction. It cannot establish independence against someone who has seen future samples; physically trusted acquisition, signature/quorum evidence, and hidden preselection remain external assumptions. It has no neural-feature generalization result. A later experimental adapter should accept a frozen activation table with precommitted source-cluster assignments, then use untouched evaluation acquisitions.

For low-context SFT critique, read REVIEW_CARD.md. Ask for one material flaw or one falsifying test, rather than a full research rewrite.
