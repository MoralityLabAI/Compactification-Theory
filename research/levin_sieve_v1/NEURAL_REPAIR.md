# LSV-04: From a learned recurrent checkpoint to a certified goal-recovery bound

**Date:** 2026-11-04. **Evidence:** Torch-trained, two-parameter CPU proxy checkpoint, exported to rational arithmetic; written global small-gain proof; exact-rational, fully-audited 128-case held-out evaluation. **Not** a pretrained TRM, a Qwen checkpoint, an empirically validated biological model, or a Lean-kernel proof.

## The learned dynamical model

A pair of recurrent components receives a forcing vector `b` and stores the mutable state `x`. The alternating asynchronous update, with state clamping on the **read** side, is

\[
    x_i(t+1)=b_i+\sum_j W_{ij}\,c(x_j(t-d_{tj})),\qquad
    c(z)=\max(-1,\min(1,z)). \tag{1}
\]

Only component `i=t mod 2` updates; others keep their previous values. Declared maximum observation ages are `D=0` (fresh) or `D=2` (stale). Histories at times `-D,...,0` are initialized to the same damaged state. The state map is nonlinear globally and is represented exactly by `fractions.Fraction`. This arithmetic is the explicit mathematical deployment contract: it does not model rounding errors of a GPU or floating-point inference kernel.

The independent task generator chooses targets `x*` with each coordinate in `[-9/10, 9/10]` and supplies

\[
 b_i=x^*_i-\sum_j A^*_{ij}c(x^*_j),\qquad
 A^*=\begin{pmatrix}0&9/5\\1/5&0\end{pmatrix}.
\]

The model observes `b` and an initial state `x(0)=x*+damage`, where each damage coordinate is at most `3/2` in absolute value. It receives **no clean target label at evaluation time**. An exact direct solver with full knowledge of the published generator is the oracle baseline. The two trainable coefficients are fitted from 512 generated labeled training pairs using Adam on the CPU, with twelve alternating cycles and intermediate losses. Evaluation uses a disjoint fixed PRNG stream; the training seed is 20261104. The weights are quantized to four decimal places once, frozen, and hashed.

| Parameter | Task generator | Trained frozen checkpoint |
|---|---:|---:|
| `W[0,1]` | 1.8000 | **1.7320** |
| `W[1,0]` | 0.2000 | **0.1845** |

Checkpoint fingerprint: `4854b2eb8ebaf51464bee325181a7fd958c2b94b1a5566bcfa8fd027fbf23ca0` (SHA-256 of canonical JSON weight map). The weight bounds built into the training parameterization enforce a stable two-node feedback product for the resulting coefficient values; this is a constrained, elementary learning experiment.

## Theorem N1: converting learned weights to global certified recovery

Let `c_j` be globally `L_j`-Lipschitz, and suppose a target `x*` obeys `|c_j(x*_j)|<=C_j`. The task generator supplies

\[
 b_i=x_i^*-\sum_j A^*_{ij}c_j(x_j^*),
\]

while the learned local update uses a fixed matrix `W`, with stale observations of age at most `D` and additional local perturbation of magnitude at most `nu_i`. Define

\[
 A_{ij}=|W_{ij}|L_j,
 \qquad \eta_i=\nu_i+\sum_j|W_{ij}-A^*_{ij}|C_j. \tag{2}
\]

Then the error `e(t)=x(t)-x*` of every local update satisfies

\[
 |e_i(t+1)|\leq\sum_j A_{ij}|e_j(t-d_{tj})|+\eta_i. \tag{3}
\]

**Proof.** Subtract the task-defining fixed point equation from the learned update and insert `W_ij c_j(x*_j)` as an intermediate term:

\[
 e_i(t+1)=\sum_j W_{ij}\left[c_j(x_j(t-d_{tj}))-c_j(x_j^*)\right]
           +\sum_j(W_{ij}-A^*_{ij})c_j(x_j^*)+noise_i.
\]

Apply the triangle inequality and the Lipschitz and clean-target envelope hypotheses. This yields (3) with `A` and `eta` from (2). If positive weights `w` satisfy `Aw<=rho*w`, `rho<1`, and every aligned block of `B` updates visits every coordinate, LSV-02 applies to (3) with `L=B(1+ceil(D/B))`. **QED.**

This proof gives a **global** bound for the deployed piecewise-linear map, because `c` is 1-Lipschitz on the entire real line. No sampled Jacobians are needed. The task and damage range enter only the target mismatch term and initial-history bound.

## Concrete frozen-weight certificate

For the actual quantized checkpoint,

\[
 |W|=\begin{pmatrix}0&433/250\\369/2000&0\end{pmatrix},
 \quad w=(3,1),\quad \rho=433/750\approx0.577333.
\]

The certificate follows from the exact inequalities `433/250 <= (433/750)*3` and `(369/2000)*3 <= 433/750`. Uniform task bounds `C=(9/10,9/10)` give

\[
 \eta=(153/2500,279/20000)=(0.0612,0.01395),\quad
 \delta=\max_i\eta_i/w_i=0.0204,
 \quad R=\delta/(1-\rho)=153/3170\approx0.048265.
\]

The known damage budget supplies the ex-ante initial weighted history error bound `H0<=3/2`; the gate **does not read a hidden target**. For `k` completed two-update cycles,

\[
 \max_i|x_i(2k)-x_i^*|\leq
 3\left[R+\rho^{\lfloor2k/L\rfloor}(3/2-R)\right]. \tag{4}
\]

- **Fresh observations:** `D=0`, `B=L=2`, so at **8 cycles** (16 coordinate updates) the right-hand side is at most `0.198551`.
- **Stale observations:** `D=2`, `B=2`, `L=4`, so at **16 cycles** (32 coordinate updates) the bound is the same.
- The asymptotic *absolute-coordinate* guarantee from this envelope is `3R=459/3170≈0.144795`, so it cannot certify an absolute tolerance of 0.1. Measured smaller errors do not change this guaranteed limit.

For the explicit piecewise-linear rational runtime, a compiler-independent checker verifies all 48 update inequalities, state histories, fair block coverage, and exact bounds on each of 128 held-out tasks under both delay settings. Negative controls intentionally mutate the weight hash, goal contract and update assumptions.

## Frozen exploratory test and matched architecture controls

`benchmark_neural.py` uses 128 disjoint held-out synthetic tasks, seed `20268207` (defined programmatically as `20261105+7102`), with a fixed generator, no selection using the test labels, and the full 24-cycle trace. Metrics are per-coordinate mean absolute error and case-level fraction with largest coordinate error at most the stated threshold. All methods receive the same task forcing and damaged initial state, and controls use the same local two-component update interface. They are **not FLOP-matched** to the direct algebraic oracle.

| Method | MAE after 8 cycles | Cases max error <=0.2 |
|---|---:|---:|
| Learned weights, fresh reads | 0.030092 | 128 / 128 |
| Learned weights, reads 2 steps old | 0.050166 | 120 / 128 |
| Untrained frozen initial weights | 0.216939 | 33 / 128 |
| Zero coupling | 0.456685 | 16 / 128 |
| Swapped learned weights | 1.083586 | 3 / 128 |

By 16 cycles, delayed learning reaches 128/128 at the 0.2 goal and MAE 0.030340. Damage alone yields MAE 0.759176, and the exact generator solver achieves zero error. Separate numerical checks include all 16 combinations of maximal target signs and maximal damage signs, for delays 0 through 3; these complement the proof and do not substitute for it.

**Practical derivative:** a fixed 24-cycle schedule can be replaced with an ex-ante certified halt after 8 fresh or 16 delayed cycles on the declared domain, a reduction of 16/24 and 8/24 cycles, respectively. These are reductions in recurrence iterations only; certificate construction, system communication, and real hardware overhead must be measured on a deployed model.

## Reproduce, provenance and scope

```bash
python -m unittest discover -s research/levin_sieve_v1 -p 'test_*.py' -v
python research/levin_sieve_v1/benchmark_neural.py --check research/levin_sieve_v1/results/repair_two_node_20261104.json
# Optional CPU torch reproduction (prints model weights; does not overwrite the frozen file):
python research/levin_sieve_v1/train_checkpoint.py --out /tmp/repaired_checkpoint.json
```

Python standard library suffices to verify the committed exact-rational checkpoint and receipt. Training additionally requires PyTorch and preserves `torch.__version__` in the source checkpoint. The run was executed on CPU PyTorch 2.10.0+cpu / Python 3.13.5. The quantized deployed weights, output metric rounding convention, seed, task/domain contract, and exact-rational simulator are committed. No GPU rental was used.

The application is intentionally limited to two trainable coefficients and an explicit, solvable task. Formalizing N1 in Lean, deriving safe influence envelopes from a real pretrained TRM checkpoint, accounting for floating-point rounding and model-output utility, and benchmarking against a nontrivial neural reference are future evidence gates. A recurrent network that has no small-gain certificate may still be useful: failure of this **sufficient condition** is not evidence of instability.
