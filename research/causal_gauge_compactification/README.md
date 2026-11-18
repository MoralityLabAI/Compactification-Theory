# CGC-01: protected causal compactification

[Mathematical theory](THEORY.md) · [Source-only audit and future protocol](PROTOCOL.md) · [Executable finite reference](finite_audit.py)

The **CGC-01** bridge takes BPD's negative controls on recurrent secret/source transport and turns them into a task-specific dimension question for compact recurrent models: *How few physically meaningful directions can explain future task behavior while respecting a precisely defined protected observer?*

For a complete recurrent state `h`, a differentiable protected output `P`, a task-output stack `T` and physical Euclidean metric, form the protected kernel `K=ker DP(h)` and stack future task Jacobians on that kernel. The singular spectrum of `DT(h)|_K` gives an **optimal local linear rank/error frontier**. A uniform task Hessian bound gives an explicit finite-radius extension. A separately derived minimum-norm correction addresses multichannel finite public-output residuals **only when** its restricted Jacobian has adequate row rank; current-readout finite equality needs an exact affine/architectural gauge.

For an exact real-affine recurrence, iterative row-span closure recovers the smallest linear state quotient preserving all declared future observations/actions. This distinguishes a locally task-informative subspace from an autonomously recurrent macrostate. Global separated-future codes impose a separate precision-bit lower bound, and nonnormal early amplification matters even when the recurrence ultimately contracts.

## Run source-only checks

```bash
python -m pip install numpy
python -m unittest discover -s research/causal_gauge_compactification -p 'test_*.py' -v
python research/causal_gauge_compactification/finite_audit.py
```

All numerical witnesses use static synthetic matrices; one independent exact rational row-rank oracle is included. **No trained BPD success, SAE discovery, model compression gain, neuron interpretation or actual secret protection is claimed.** Theoretical identities concern explicitly stated physical metrics, output weightings, protected channels and task horizons.

## Crosswalk

- [BPD source/intervention experiments](https://github.com/MoralityLabAI/BPD): SF03's signed-gradient null, SF04 source-binding failure, SF05 finite versus tangent public repair, SF05B source-closure training tradeoff. The full physical BPD matrix and gauge-invariant blocks are the candidate units, subject to heldout causal validation.
- [RC-01](../recurrent_circuits_v1/THEORY.md): resource-bounded compactness and exact affine `F_2` quotient. CGC-01 adds a real-linear controlled version and protected task-conditioned tangent analysis.
- [LSV-15](../levin_sieve_v1/pretrained_trm/GAUGE_QUOTIENT.md) / [LSV-18](../levin_sieve_v1/pretrained_trm/LSV18_SEMICONJUGACY.md): exact LayerNorm symmetry and future-closed latent state criteria.
- [NOR-04/10](../nor01_numerical_orbit_sketch/NOR04.md): source-locked block contraction and the two-basin terminal-bit obstruction. Local tangent rank and global basin-label bits are different.
- [SMC-15](../real_recurrence_sae_v1/SMC15_RESULTS.md) / [SCM-15](../dimensional_atlas/SCM15_THEORY.md): output-near-null SAE coalitions, tangent atlas and physical frame comparisons.
- [Precision frontier](../precision_entropy_frontier/README.md): task-separated futures and quantized code/parameter cost.

A future neural experiment should be registered as a separate CGC/BPD-GM follow-up with new source-family and independent model-root evaluation. CGC-01 remains a **source-only mathematical addition**.
