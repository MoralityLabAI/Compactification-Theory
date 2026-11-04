# LSV-16: Transporting ambient recurrent-state damage into a goal-visible LayerNorm quotient

Date: 2026-11-04. Proof class: analytic inequalities plus computationally rigorous ball-arithmetic bounds on a separately trained, pinned 44,001-parameter Sudoku recurrence. LSV-15 established goal robustness *at* the last LayerNorm bus. LSV-16 asks if stage-one damage can be transported to that bus without assuming any particular perturbation direction. The exact-real model interpretation and frozen stage-one roster are unchanged.

## Theorem ATG-01 (componentwise sound transport)

Suppose a clean vector `x` lies in an exact-real computation graph and a perturbed vector `x+e` has `|e_i|<=b_i`. Affine layers with weights `W` transport the error by `|W|b`, independent of the input value. Standard erf-GELU is globally `5/4`-Lipschitz: its derivative is `Phi(x)+x phi(x)`, and

`|Phi(x)+x phi(x)| <= 1+1/sqrt(2*pi*e) < 5/4`.

For LayerNorm applied to `d` coordinates, let `u=Px=x-mean(x)1`, `s=sqrt(eps+||u||_2^2/d)>0`. Define the computable componentwise bounds

`c_j = (1-1/d)b_j + (1/d) sum_(k!=j) b_k`, `R=sqrt(sum_j b_j^2/d)`.

Then `|P e|_j<=c_j` and `||P e||_2/sqrt(d)<=R`. If `R<s`, the exact-real LayerNorm with frozen `(gamma,beta)` satisfies

`|LN_j(x+e)-LN_j(x)| <= |gamma_j| [ c_j/(s-R) + |u_j| R/(s(s-R)) ]`.

**Proof.** The first bound follows by expanding `e_j-mean(e)` and using the triangle inequality. Orthogonal projection does not increase Euclidean norm, which proves the second. Regard `s` as the Euclidean norm of the augmented vector `(Px/sqrt(d),sqrt(eps))`. The reverse triangle inequality gives `|s'-s|<=R`, hence `s'>=s-R>0`. Split the difference of normalized coordinates as `(u_j+Pe_j)/s' - u_j/s = (Pe_j)/s' + u_j(1/s'-1/s)` and apply the two bounds. Multiplying by `|gamma_j|` proves the inequality. □

**Corollary.** Applying affine, erf-GELU and LayerNorm transport through the fixed three remaining recurrent blocks gives a sound componentwise enclosure of the error at the *input* of the fourth block's final LayerNorm. Center each affected token's error by `P`, then compute `r_goal = max_j c_j` at the selected goal cell. If `r_goal` lies strictly below the exact LSV-15 per-case sufficient transverse threshold `T_goal`, that case's predicted top digit is provably unchanged under **every** stage-one perturbation in the declared ambient infinity box. The same test against the LSV-15 runner-up threshold certifies the top-two order. Neither claim establishes that the prediction is a correct Sudoku solution.

## Computational implementation

[`transport_gauge.py`](transport_gauge.py) computes *clean center values in Arb* using the frozen source-locked checkpoint and encloses deviations using the theorem. All nonnegative error radii are rounded outward to Arb endpoints after linear operators. The LayerNorm bound uses verified lower estimates of the clean variance and verified upper bounds on centered input deviations. It reports failure if a denominator lower bound becomes nonpositive. Selected clean centered states must overlap the previously serialized and independently replayed LSV-15 witness intervals.

[`run_transport_gauge.py`](run_transport_gauge.py) reuses the original exact Sudoku roster and compares the derived final transverse radii against the archived independent LSV-15 exact-rational limits. It keeps failure cases and the full attempted radius roster. No new training, label-based selection, or post-hoc intervention change is made. Exact floating-point device implementation error and the pretrained model's actual task correctness are outside this theorem's scope.

## Falsification and limitations

An observed larger radius may result from a stronger proof bound or a different intervention location. A failure to certify means the enclosure is inconclusive. The theorem uses globally valid GELU bounds and a variance-dependent LayerNorm inequality; its radius may grow substantially through token mixing, so `1e-8` is an *experimental target*, not a guaranteed outcome. This approach is simpler than affine/Taylor arithmetic but preserves the task-relevant LayerNorm centering structure that plain interval boxes lose.
