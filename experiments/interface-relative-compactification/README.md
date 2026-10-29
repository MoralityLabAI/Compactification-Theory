# IRC-02: interface-relative compactification experiment protocol

**Status:** prospective, documentation-only protocol, 2026-10-29. No new model training, native StarCraft runs, human-interface measurements, or BCI tests have been performed by this addition.

## Canonical mathematics and ownership

EvalMath owns the exact mathematical reference, finite Bayesian oracle, test suite and source-hashed receipt, reviewed in [EvalMath PR #21](https://github.com/MoralityLabAI/EvalMath/pull/21), branch `research/irc02-timed-predictive-quotient-20261029`, pinned head **`95ff27ac746f2883108e925888da05a8e42d81b6`**. Read [the proof and limitations](https://github.com/MoralityLabAI/EvalMath/blob/95ff27ac746f2883108e925888da05a8e42d81b6/paper/interface_relative_compactification.md). The pinned reference treats rewards as observable feedback, uses positive integer durations, and keeps all action legality public.

This repository owns the resource-accounting and model experiment questions. The runnable [Pluto-Gym](../../research/pluto_gym/README.md) already owns synthetic multi-front scheduling, compact student training and rule-designer curriculum comparisons. Its exact reachable reward ceiling may serve as an independent **scheduler-specific** oracle. IRC-02 adds a formal stochastic *history-equivalence* contract and adversarial representation tests; it does not replace Pluto-Gym or vendor EvalMath code.

Ownership exclusions: [HC-01 PR #1](https://github.com/MoralityLabAI/Compactification-Theory/pull/1) owns learned H-Net chunking and the deterministic behavioral quotient; [RC-01 PR #2](https://github.com/MoralityLabAI/Compactification-Theory/pull/2) owns recurrent-circuit GPU/host-DRAM runs; [Plan R PR #3](https://github.com/MoralityLabAI/Compactification-Theory/pull/3) owns its bounded-power deterministic timed word oracle; [IC-01](../interventional-compactification/README.md) owns existing intervention/activation work. This protocol changes none of their code, tests, or frozen criteria. Native [TensorCraft PR #5](https://github.com/patrickdugan/TensorCraft/pull/5) remains owned separately.

## Question and mathematical contract

Given an interface `I`, a public mode `q` (for example, camera and selection), and a remaining deadline `d`, which distinctions between observable histories must survive compression to predict **every legal future action/observation/reward/time trace**?

EvalMath's exact reference computes the signature of a posterior history under legal feedback-adaptive future policy trees. For a declared finite set of histories at each `(q,d)`, let `N(q,d)` count predictive equivalence classes. With public mode and deadline delivered separately, exact fixed-length memory needs

\[
M_{\min}=\left\lceil\log_2\max_{q,d}N(q,d)\right\rceil.
\]

That is an *instance-state cardinality* bound for a specified finite history domain. Model parameters, code/lookup tables, input caches, policy-tree descriptions, clock delivery, and training data are separate resources. It is not a universal statement that every POMDP with `|S|` hidden states requires `log2 |S|` bits.

The experimental resource vector is

\[
\mathcal R=(D_\theta,B_{\rm live},C_{\rm infer},T_{\rm passes},
 Q_{\rm observation},C_{\rm command},C_{\rm switch},C_{\rm train},
 M_{\rm host},M_{\rm GPU},\varepsilon).
\]

`D_theta` includes learned weights, encoder/decoder, externally selected rule tables and instruction programs. `B_live` charges retained history, recurrent state and cached observations. Interface observation capacity, command latency and screen-switching delay are independent experimental variables, expressed with concrete measurement units.

Define optimal finite-interface value `V_I*` by exact dynamic programming when the declared test world supports it; compare trained policy value `V_I(pi)` and regret `V_I* - V_I(pi)`. For probabilistic forecasts also record rational calibration on exact finite distributions. Preserve the difference between **prediction sufficiency for every legal continuation**, **optimal-policy sufficiency**, and **mean sampled performance**.

## First bounded phase: exact finite oracles

1. **Pin source and inputs.** Fix the EvalMath commit and source hashes; assign immutable fixture IDs to `S`, `Q`, public action grammar, kernel probabilities, public reward/termination, duration distributions, candidate belief histories, horizon and seed. Record the entire event tuple `(signal,reward,duration,next_mode,done)`.
2. **Run exact oracle.** Compare recursive predictive signatures with independent hidden-state cylinder enumeration and exhaustive policy trees for small instances. Check the minimum fixed-bit cardinality on the **declared history set**. Replay the returned shortest distinguishing policy within its deadline.
3. **Reproduce counterexamples.** Value-only collisions; two-tick delayed observation; camera switch; zero-probability feedback; mixed terminal/duration outcomes; deterministic four-state hold/swap reduction. Verify that expanding a permitted action grammar refines exact all-futures partitions when all other semantics are unchanged.
4. **Mutation gates.** Deliberately alter probability mass, next-state branch, duration or public action access. Recompute the oracle and require a changed witness or clearly declared equality where the altered parameter is observationally irrelevant. Re-run normal and optimized verification. A source hash mismatch is a failed receipt, never a reason to silently replace a baseline.

Reproduce the pinned EvalMath module from an independently checked checkout:

```bash
cd ../EvalMath
# Verify HEAD matches 95ff27ac746f2883108e925888da05a8e42d81b6
PYTHONPATH=src python -m unittest discover -s tests -p test_interface_relative_compactification.py -v
python scripts/check_receipts.py --only irc02
```

For an oracle import, prefer an exact revision and separate environment; avoid copying `src/spiral_ln` into this repository.

## Second phase: parameter, time and interface frontier

At each interface setting, hold the same rules, starting-state distribution, opponent policy, latent events and reward definition. Sweep parameter storage bits, mutable memory bits, inference passes, observation window, addressable action set, and switching/command time. Record clock and camera state as actual inputs. A student with the same policy network but a broader raw API can be more successful because of **interface accessibility**; an inaccessible interface ceiling is determined by the corresponding exact oracle.

Start with the existing synthetic Pluto-Gym environment, which already provides frozen, random-curriculum and designer-curriculum baselines. Use IRC-02 first as an **oracle compatibility test**, with a separately declared translation from Pluto-Gym state/action/reward timing to the rational finite kernel. Confirm agreement of both exact values on common small cases. Pluto-Gym's currently unmetered schedule search must be accounted for as independent compute before making resource-efficiency claims. Its existing synthetic task does not, by itself, exhibit partial observability or an exact posterior-history quotient; additional game dynamics require a separate reviewed change.

Use group-balanced held-out rule families and paired seeds. Freeze training updates, collected samples, optimizer steps, decoder capacity and teacher call budgets. Compare learned memory against an exact class-index upper control, a memoryless policy, a replay-driven policy, a shuffled-state code and an oversized code. When a larger designer changes games, treat designer output, feedback, model revision and mutation grammar as tracked resources. Hold final oracle cases blind to designer and student selection.

### Evidence and acceptance

| Claim | Minimum evidence |
|---|---|
| Exact quotient on finite fixture | Equality of every declared independent cylinder law; exact source-pinned receipt |
| Sufficient learned state on held-out test distribution | Calibration and regret on independently generated held-out histories; paired seeds and confidence intervals |
| Uniform future-trace guarantee | Finite exhaustive closure or a separately proved error/TV bound covering all reachable conditional histories |
| Interface restriction on a measured human | Human control trials with recorded switch latency, error rates, and observation availability; calibrated fit to the specific cohort |
| Superhuman-by-interface demonstration | Measured performance exceeding a **proved** upper bound for a specified narrower interface, at matched game conditions |

**Failure witnesses:** two histories given the same code that have different allowed future trace distributions; a learned decoder whose cost was omitted; an action whose legality varies secretly with hidden state; or a policy clock with uncharged external memory. If the exact kernel cannot be constructed from a model environment, report only empirical comparisons and mark the theorem bridge unverified.

**Freeze:** no further E10 selector runs, H-Net training, RC-01 GPU/DRAM experiments, unapproved designer network requests or native StarCraft execution. Real BCI work belongs to a separate approved human-subjects research protocol. IRC-02 is mathematical and synthetic until such data exist.
