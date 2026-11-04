# LSV-15: A correlation-preserving goal quotient at the recurrent LayerNorm bus

**Date:** 2026-11-04. **Mathematical evidence:** an exact LayerNorm symmetry and a nonasymptotic rational margin bound. **Model experiment:** source-locked, externally pretrained 44,001-parameter TRM-inspired Sudoku recurrence. The intervention bus in this note is the **input to the final LayerNorm in the fourth recurrent update**. The theorem applies to any such LayerNorm in the architecture, and the quantitative checkpoint experiment specifies one bus. This is a *distinct perturbation geometry* from the LSV-14 radius `1e-14` around all 3888 initial stage-one coordinates.

## The quotient symmetry

Let a token have feature dimension `d=48`. Write

\[
P=I-\frac{1}{d}\mathbf1\mathbf1^\top,\quad
u=Py,\quad s(y)=\sqrt{\epsilon+\|Py\|_2^2/d},\quad
N(y)=\gamma\odot\frac{Py}{s(y)}+\beta .
\]

Here epsilon is the **exact binary64 number** used as PyTorch LayerNorm's epsilon, while gamma and beta are frozen exact binary32 weights embedded in the reals. The implementation reads and checks the full upstream checkpoint with the existing LSV-05 hash lock.

**Theorem GQ-01 (unbounded gauge immunity).** For every `y in R^d`, any scalar `a in R`, and any epsilon greater than zero,

\[
N(y+a\mathbf1)=N(y).
\]

**Proof.** `P1=0`, hence `P(y+a1)=Py`. The normalizing denominator, numerator and frozen affine parameters coincide. QED.

The product over 81 tokens yields **81 independent unbounded common-shift interventions** that are invisible to the following LayerNorm, and thus invisible to all later recurrent blocks and final logits if the intervention is injected exactly at this bus. This result is structural and requires neither a local Jacobian estimate nor a small perturbation. It is the familiar shift invariance of LayerNorm, applied here as an operational task quotient. No new general theorem is claimed.

## The transverse bound

A general perturbation is `delta=a*1+v` with `Pv=v` and `||v||_infty<=r`; the arbitrary scalar `a` drops out. Let `u=Py`, `s0=s(y)>r`. Since `s(y)` is the Euclidean norm of the augmented vector `(Py/sqrt(d),sqrt(epsilon))`, the reverse triangle inequality gives

\[
|s(y+\delta)-s_0| \le \|v\|_2/\sqrt d \le r.
\]

Therefore, for each normalized output feature,

\[
\boxed{ |N_j(y+\delta)-N_j(y)|\le
|\gamma_j|\frac{r}{s_0-r}\left(1+\frac{|u_j|}{s_0}\right)=:d_j.}
\]

**Proof.** Use `u'=u+v`, `s'=s(y+delta)>=s0-r` and

\[
\left|\frac{u'_j}{s'}-\frac{u_j}{s_0}\right|
\le\frac{|v_j|}{s'}+\frac{|u_j||s'-s_0|}{s's_0}.
\]

Apply the displayed lower bound on `s'` and multiply by `|gamma_j|`. QED.

For the affine head `ell_k=\sum_j W_{kj}N_j+b_k`, fix a clean winner digit `a`. A strict, checkable sufficient condition for `a` to remain the winner under **every gauge shift and every transverse perturbation with `||Pdelta||_infty<=r`** is

\[
\boxed{\ell_a(y)-\ell_k(y)>\sum_j|W_{aj}-W_{kj}|d_j
\quad\text{for every }k\ne a.}
\]

The proof subtracts the two logits and applies the above featurewise bounds. For preserving a designated runner-up, check its gap against every digit other than the winner.

### Machine-checkable rational witness

The verifier records 48 rigorous Arb enclosures of the **centered final pre-LayerNorm input** at the selected Sudoku goal cell. The state is reached by exactly three further recurrent passes from the original hash-locked stage-one tensor. These balls are serialized with a check that every decimal interval contains its originating Arb object. The witness also records exact hexadecimal binary floats for gamma, beta, epsilon, head matrix and head bias.

The standalone Python `Fraction` checker:

- Computes lower and upper bounds on the clean variance from the centered interval entries.
- Uses integer square roots to enclose `s0` by rational endpoints.
- Reconstructs all ten clean output logits using interval arithmetic.
- Checks the winner, runner-up, and every perturbed winner-versus-competitor gap with a rational strict inequality.
- Recomputes reported counts and rejects changed source/checkpoint identities and goal indices.

The trusted numerical component is the Arb reachability computation that supplies the 48 initial centered enclosures; a tampered receipt could substitute invalid intervals and therefore requires hash/source-controlled creation in CI. The rational replay checks its arithmetic independently.

## Experimental limits

This certificate concerns **interventions at the final LayerNorm bus**, with tokenwise shared shifts treated as correlated variables. It also yields an exact-repair result for any earlier recurrence bus of the same form, because the shift is removed at the next LayerNorm call and future deterministic computation receives the same state. Ordinary FP32/FP64 hardware arithmetic can incur cancellation errors for extremely large shifts; no hardware-wide infinite-amplitude guarantee is implied.

This is evidence about a goal-invisible direction in a recurrent architecture. It does not prove larger full-dimensional stage-one neighborhoods, new Sudoku skills, robustness to arbitrary hidden-state edits, or a stronger rank-four result than LSV-14. A future proof effort should combine this analytic quotient with high-dimensional affine/Taylor enclosures for the remaining `d-1` transverse components per token, aiming at an ambient-stage-one radius of `1e-8`.

## Reproduction

```bash
python -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p 'test_gauge*.py' -v
python research/levin_sieve_v1/pretrained_trm/run_gauge.py --fetch --max-cases 64 --out runs/lsv15/gauge.json
python research/levin_sieve_v1/pretrained_trm/verify_gauge_receipt.py runs/lsv15/gauge.json
```

The initial first-missing-cell pair and test roster were fixed in LSV-13. Independent novel mathematical discovery is evaluated separately by the philosophy-sieve protocol.

## Theorem GQ-02: Exact quotient dimension with positive epsilon

Assume every entry of `gamma` is nonzero. Then LayerNorm identifies **precisely** the equivalence classes `y ~ y'` iff `y-y'` is a multiple of the all-ones vector. In particular, its Jacobian has rank exactly `d-1` at every input; its kernel is `span{1}`.

**Proof.** Set `v=(N(y)-beta)/gamma=u/s`, and `q=||v||_2^2/d`. Then `q=t/(epsilon+t)` where `t=||u||_2^2/d`, so `0<=q<1` and

\[
  u=v\sqrt{\frac{\epsilon}{1-q}}.
\]

Thus the normalized output uniquely recovers the centered input `u`. Two inputs have the same normalized output exactly when their centered inputs agree. The kernel of the projection `P` consists of shared translations.

Differentiating `u/s` on the centered subspace gives `I/s - uu^T/(d s^3)`. Its eigenvalues are `1/s` in the `d-2` centered directions perpendicular to `u`, and `epsilon/s^3>0` in the direction of `u` (or `1/s` when `u=0`). Multiplication by invertible diagonal `gamma` preserves rank. The missing one dimension is the translation direction. QED.

The frozen checkpoint has **48/48 nonzero gamma parameters** in its last LayerNorm. Its complete 81-token LayerNorm therefore has an exact **81-dimensional gauge kernel**, with rank `81*(48-1)=3807` on the 3888-dimensional pre-normalization state. This is a known consequence of LayerNorm geometry with positive epsilon, now confirmed for the specified learned weights.

## Theorem GQ-03: Direct radius optimization for the certified bound

Let `S>0` be the rational lower bound on the clean `s0`, let `U_j` bound the absolute centered input, and let `m_ak>0` be a rational lower bound on the clean winner-versus-competitor logit margin. Set

\[
B_{ak}=\sum_j|W_{aj}-W_{kj}|\,|\gamma_j|(1+U_j/S).
\]

The transverse proof above has sufficient condition `m_ak > B_ak*r/(S-r)`. Its **largest sufficient open radius for this particular inequality** is

\[
\boxed{T_{ak}=\frac{S\,m_{ak}}{m_{ak}+B_{ak}}.}
\]

Indeed, rearranging the inequality with `0<=r<S` gives exactly `r<T_ak`. A winner remains certified for `0<=r< min_{k!=a}T_ak`. To preserve the designated runner-up as well, take the minimum with `min_{k notin {a,b}} T_bk`. If `B_ak=0`, the formula gives `T_ak=S`, although the true shift-invariant margin can hold at any radius. These thresholds optimize the *chosen analytic sufficient bound*, not the actual adversarial radius.

The independent exact-rational [threshold verifier](gauge_threshold.py) recomputes these inequalities from the archived source-locked Arb enclosures and hex-exact weights. The [64-case threshold record](results/lsv15_exact_thresholds_64_20261104.json) proves that **every selected top digit is stable for transverse radius strictly less than `0.07546051515`**, plus arbitrary common-mode shifts. For the designated top two digits, the corresponding all-case strict radius is `0.00140648959`. The original grid experiment also certified all 64 winners at radius 0.01 and 58/64 second-ranked digits there. The exact rational thresholds have been retained without rounding in the JSON record.
