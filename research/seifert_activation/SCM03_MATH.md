# SCM-03: nonabelian braid structure in recurrent activation traces

**November 13, 2026.** Synthetic exact-algebra instrument and post-hoc RMP capture extension. No trained neural result, 4D embedded-surface isotopy claim, or Miller-type Khovanov map is asserted.

## Pairwise winding loses commutators

For n distinct labeled points in a fixed planar feature chart, the fundamental group of the configuration space is the pure braid group P_n. Its abelianization is Z^(n choose 2), and the integer coordinates are pairwise winding numbers. Once n>=3, pure braid commutators may have vanishing pairwise winding and still be nontrivial. The previous SCM-02 two-token default was inherently abelian; SCM-03 captures three downstream tokens by default.

Set A = sigma_1^2 and B = sigma_2^2 in P_3. Let C = ABA^-1B^-1. The braid words e, C and C^-1 all have pairwise winding (0,0,0). Exact Artin action on the free group F_3 distinguishes e from C and distinguishes C from C^-1. C and C^-1 also share the normalized one-variable Alexander polynomial (t-1)^4 and canonical Seifert-algorithm genus 2.

These are distinct BASED braid histories, with labeled starting configurations. Different Artin images do not establish that closed links differ: braid conjugation and Markov stabilization can preserve link isotopy while changing the based action. Nothing here distinguishes equal-genus Seifert surfaces in B^4. That would require separate properly embedded surfaces and surface-sensitive invariants.

## Faithful exact Artin action

Represent a word in F_n as signed generator indices; free-reduce by canceling adjacent inverses. Generator sigma_i induces:

    x_i -> x_i x_(i+1) x_i^-1
    x_(i+1) -> x_i

Its inverse induces:

    x_i -> x_(i+1)
    x_(i+1) -> x_(i+1)^-1 x_i x_(i+1)

Other generators remain fixed. Composing these automorphisms computes a faithful representation of the based braid. The test suite checks braid relations, commuting distant generators, inverse cancellation, a known nonabelian commutator, 16 sign/strength variants, and a conjugacy counterexample. This calculation is exact over finite free words, subject to a configurable image-size cap.

An Artin image tuple is an exact comparison when the same labeled basepoint is used. Identical image tuples are meaningful for the same deterministic cycle captured at 32 and 64 angles; changing the chart or intervention radius may move the basepoint, so comparison requires an explicit identification or should use closure invariants. SHA-256 is retained as a receipt fingerprint, not a formal equality proof. For short words the complete free-group images are recorded and compared. A size-budget failure yields an explicit inconclusive status.

## Compactification hypothesis

A code containing only pairwise winding cannot recover the full braid class. Adding the one-variable Alexander polynomial still collapses the C versus C^-1 example, even though the based Artin actions differ. If a finite set has K causally distinguishable based-braid classes sharing those simpler summaries, exact fixed-length classification of those classes needs at least ceil(log2 K) bits. The existence of different topological histories alone does not show that the model requires distinguishing them.

A possible cheaper hierarchy uses the lower central quotients P_n/Gamma_k(P_n) between full Artin images and the abelianization. Compare information retained against exact Artin classes on the same synthetic words before attempting a learned compact code. Meter the cost of free-word expansion and treat budget failures as missing evidence.

## Neural pilot

Run RMP LOOP-T and FF-U on the existing unsealed monitor_calibration selection half, using group-disjoint examples and frozen D-shared chart and intervention directions. Leave G0, existing registrations, dictionaries, checkpoints and reporting/holdout splits unchanged. Inject at position 4; track downstream positions 16, 32, and 48 across eight visits. Capture 32 and 64 equally spaced intervention angles with exact closure checks. The valid sample path must be collision-free. The current polygonal measurements make no claim about unobserved continuous arcs without externally proved derivative bounds.

Compare exact Artin actions between angular resolutions when the images fit the cap. Score every rejected/indeterminate trace, report nonabelian witness incidence, crossing counts, clearance, model calls, effect on the named reasoning variable and final predictions. Chart effects require random orthogonal, PCA and norm-matched controls selected without looking at the held-out reporting half. Later compression comparisons should use SCM-02's relative-error homotopy certificate and separate causal intervention checks.

**References:** Jenny Wilson, Five definitions of the (pure) braid group (https://ncatlab.org/nlab/files/Wilson-BraidGroups.pdf); Wilson, Representation Stability lecture notes (https://dept.math.lsa.umich.edu/~jchw/FILectures.pdf); classical Artin action of the braid group on a free group. SCM-01 and SCM-02 supply the braid extraction, Alexanders and sampling protocol.
