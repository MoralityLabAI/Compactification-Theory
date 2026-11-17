# SCM21–BPD observable-mask bridge

Read-only regression check for the public SCM21 conservative-causal experiment after PR #207 merged. The companion exact mathematics is in [`MoralityLabAI/BPD` PR #14](https://github.com/MoralityLabAI/BPD/pull/14), `mechinterp/bpd/theory/causal_observable_masks/`.

From Compactification-Theory repository root:

```bash
python -m unittest discover -s research/dimensional_atlas/gym/bpd_quotient_bridge/tests -v
python -m research.dimensional_atlas.gym.bpd_quotient_bridge.audit --check \
  --out research/dimensional_atlas/gym/bpd_quotient_bridge/results/public_scm21_causal_quotient.json
```

The bridge uses only `scm21_conservative_causal/results/summary.json` and checks its **Git blob SHA**. It reconstructs published correct counts, common-eligible denominators, weighted intervention norms, and a lower bound on direction-specific successes from aggregate marginals. It makes no per-pair outcome inference, does no neural replay and leaves private checkpoints and protected EvalMath test sets alone.

**Result:** at zero-based visit 1, native VPD write was 0/768, difference-PCA 147/768, head-readable 413/768. Difference-PCA patches were about eleven times larger by mean L2 dose. Thus this source shows a sharp *combined direction/dose* effect rather than isolating a causal basis advantage.

BPD's new theorem identifies observational-equivalence masks by a time-dependent Gramian, motivating a prospectively registered, per-pair dose-matched native BPD edit comparison across visits. A sparse basis may be underidentified by its reconstruction objective even when the observable causal effect is identified after sufficient recurrence.
