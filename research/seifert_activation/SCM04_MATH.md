# SCM-04: Closure-invariant Milnor linking and finite-degree compactification

**Status:** exact discrete algebra and synthetic/mock execution only. The link belongs to a frozen 2D activation readout chart, not intrinsically to the high-dimensional neural state space. The module computes *an invariant of an oriented, ordered closed three-component link*. It does not compute 4D Seifert-surface isotopy or Khovanov cobordism maps.

## 1. Why a closure-level observable matters

SCM-03 computed Artin's faithful action on the free group and distinguished the based braids `C` and `C^-1`, where

    A = sigma_1^2, B = sigma_2^2,
    C = A B A^-1 B^-1.

A based action can change after an angular cut (braid conjugation), even if the closed link stays fixed. Pairwise winding of `C` and `C^-1` is `(0,0,0)`. They have the same one-variable Alexander polynomial `(t-1)^4` and canonical Seifert-algorithm genus 2. Their **oriented, ordered closures**, however, have *distinct* Milnor triple-linking values, `+1` and `-1` with the convention below. The sign depends on component orientation and ordering; an unoriented or unlabeled comparison may identify these two presentations.

For a pure 3-braid `beta`, Artin's action has `phi(x_i) = u_i x_i u_i^-1`. Choose the unique `u_i` whose total exponent of `x_i` is zero. Under Magnus expansion `theta(x_j)=1+X_j`, `theta(x_j^-1)=1-X_j+X_j^2+...`, the coefficient

    mu_123(beta) = [X_1 X_2] theta(u_3)

is a length-three Milnor number. Its cyclic permutations `[X_2 X_3] theta(u_1)` and `[X_3 X_1] theta(u_2)` agree modulo the usual indeterminacy

    d = gcd(lk_12, lk_13, lk_23).

For `d=0` they agree as integers, and the number is an isotopy invariant of the **oriented, ordered link closure**. With `d>0`, the residue modulo `d` is the link invariant. The implementation computes the actual pairwise linking numbers independently by tracking crossing strand identities. The exact-longitude implementation uses cyclic free reduction of the Artin images and degree-two Magnus coefficients. A separate noncommutative polynomial reference computes the same coefficient in the tests.

References: Milnor, *Link Groups* (Annals of Mathematics, 1954); standard Magnus/Fox-derivative and Artin-action definitions; *Arithmetic Topology in Ihara Theory*, Publ. RIMS 53 (2017), pp. 629–688, DOI: 10.4171/PRIMS/53-4-6, https://ems.press/journals/prims/articles/15057 ; *Pure braid groups, Milnor invariants, and Johnson homomorphisms*, https://catalog.lib.kyushu-u.ac.jp/opac_download_md/1806826/math0211.pdf .

## 2. Cubic Magnus quotient is sufficient when pairwise linking vanishes

Expanding full `phi(x_i)` in reduced free-group words can cause exponential growth in the braid word length. We can compute the closure's **exact** Milnor triple linking using only Magnus series of degree at most three.

Let `I` be the augmentation ideal of `Z<<X_1,X_2,X_3>>`. Work modulo `I^4`. For a braid word, propagate its three Artin-generator images by truncated noncommutative multiplication; invert `1+q` using the exact finite identity

    (1+q)^-1 = 1-q+q^2-q^3   (mod I^4).

Each image requires at most `1+3+9+27=40` integer coefficients. When all pairwise linking numbers vanish, each normalized longitude lies in the free commutator subgroup, hence has expansion `theta(u_i) = 1 + U_i^(2) + O(I^3)`. Therefore

    theta(phi(x_i)) = (1+U_i^(2))(1+X_i)(1-U_i^(2)) + O(I^4)
                    = 1+X_i+U_i^(2) X_i-X_i U_i^(2)+O(I^4).

For three distinct indices `a,b,i`, the coefficient of `X_a X_b X_i` is precisely `[X_a X_b] theta(u_i)`. In particular,

    mu_123 = [X_1 X_2 X_3] theta(phi(x_3))
           = [X_2 X_3 X_1] theta(phi(x_1))
           = [X_3 X_1 X_2] theta(phi(x_2)).

This computes an exact integer with **constant coordinate dimension per strand** and work linear in braid length for fixed n=3 and polynomial degree 3. Integer coefficient *bit lengths* grow with braid complexity. This is a finite nilpotent quotient, rather than a guarantee of constant bits or a faithful embedding of arbitrary braids. Two distinct braid closures can share the entire degree-three quotient.

The fast implementation checks purity and zero pairwise linking before applying the cubic shortcut. Nonzero-linking cases use the full longitude and report `mu_123 mod d`. For long examples with nonzero linking, a word-budget error is reported explicitly.

## 3. Angular phase-cut correction

A subtle failure emerged in the executable: changing the start angle reorders the **x ranks** of token-labeled strands. The braid extractor's Artin letters use the x-position order at the chosen start frame. For a three-component link, an odd permutation of component labels reverses the sign of `mu_123` (or its residue modulo `d`).

Let `pi = argsort(z_i^x(theta_0))`, sending the x-rank to the original tracked token index, and define `epsilon(pi)=(-1)^inversions(pi)`. The canonical token-ordered invariant is

    mu_123^tokens = epsilon(pi) * mu_123^x-rank   (mod d, if d>0).

A 25-phase-cut synthetic test on the commutator fixture changes the *raw* value between `+1` and `-1`; its corrected token-order invariant remains `+1` at every cut. This is a measured-chart coordinate correction. Flipping a chart axis's orientation or relabeling tokens changes conventions and must be registered before comparing datasets.

## 4. Compactification and what it certifies

Consider the family `C^k` for integer `k`. Its pairwise linking numbers vanish and `mu_123(C^k)=k`. The cubic quotient computes `k=+1000` and `k=-1000` with bounded polynomial-coordinate count, while expanding full free-group words fails the configured 20,000-letter budget for much smaller `|k|`.

For the finite family `k=-M,...,M`, any deterministic fixed-length code that must **exactly recover the realized triple invariant** has at least `2M+1` codewords, so needs at least `ceil(log2(2M+1))` bits. For `M=1000` this is 11 bits. It is a lower bound on preserving *this specified observable*. It neither says that the neural model needs this observable nor bounds compression needed for other tasks. At fixed linking **and** fixed Alexander polynomial, the pair `C,C^-1` already proves that an Alexander-only code loses one binary distinction, subject to oriented labels.

The SCM-02 pair-relative-error certificate preserves the full observed PL braid class when its sufficient margin holds, which automatically preserves `mu_123`. Failing that margin does not establish a topology change. The new invariant is a cheaper, less expressive necessary consistency check on compression when a full braid certificate is unavailable.

## 5. Excluded claims and next empirical gate

- No trained RMP `LOOP-T`/`FF-U` checkpoint, D-shared dictionary or monitor bundle is accessible from this work environment. The CPU tests exercise synthetic braid words and a mock visit-hook model.
- No invariant here determines equivalence of two **Seifert surfaces with the same link**; Miller-type same-boundary, same-genus non-isotopy remains a later four-dimensional problem.
- Nonzero Milnor values would measure topology only in a **specified projected chart**. Statistical validity and causal relevance require frozen charts, paired interventions, angular refinement, phase-cut checks, independent controls, and held-out causal outcomes.
- For the neural phase, compare the added `milnor.mu123_token_order` at each recurrent visit, and test whether it predicts task-relevant counterfactual effects beyond linking numbers, one-variable Alexander and randomized chart controls. Freeze the SAE axes and all evaluation splits before running.
