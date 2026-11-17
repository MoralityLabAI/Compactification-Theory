# SCM-21: Rank-four causal-subspace capacity audit (CPU)

**Execution class:** exact retrospective audit of SCM-20's *public aggregate* causal-intervention receipts, plus a separately labeled synthetic linear-algebra control. The underlying SCM-20 trained checkpoints remain on the RTX 3050 host; SCM-21 runs no neural inference, training, or GPU computation. This code is deliberately standalone inside `Compactification-Theory`.

The question is whether the SCM-20 native VPD intervention's very low success rate can be explained by an unavoidable four-dimensional rank bottleneck. SCM-20 already tested several rank-four projections against *identical* clean-eligible donor-recipient pairs, leaving a directly auditable partial answer. The reported comparison was selected after seeing the receipts and is **retrospective**.

## Run in the repository checkout

```sh
python -m pip install numpy==2.3.5
python -m unittest discover -s research/dimensional_atlas/gym/scm21_capacity/tests -v
python research/dimensional_atlas/gym/scm21_capacity/analysis.py \
  --source research/dimensional_atlas/gym/scm20_causal_consistency/results/summary.json \
  --check
python research/dimensional_atlas/gym/scm21_capacity/synthetic.py --check
```

`analysis.py --source` checks the exact Git blob SHA-1 of the previously committed SCM-20 public summary and independently reconstructs the six-cell rank-four counts and clean results. The portable compact snapshot is separately checked for format and coherent denominators. For an offline environment without a complete checkout, omit `--source` to replay the already extracted data, and explicitly report that the upstream bytewise comparison was unavailable.

The check commands are **read-only**. Regenerate the two reported CPU receipts by running the scripts without `--check`. Synthetic `synthetic.py` depends on NumPy only; the published SCM-20 count analysis is pure standard-library Python.

## What is established

On the SCM-20 causal model and at visit 2 (zero-based index 1), the *rank-four difference-PCA* projection has 217/748 successes across the six common-eligible seed/split assays. The *rank-four native write-span projection* has 63/748. A sharp marginal bound forces at least 154 difference-PCA-only successes, even without the unavailable per-pair cross-method contingency table. The native learned-write endpoint scores 55/748.

This refutes the claim that **every rank-four patch in the existing test system must be as ineffective as the native VPD write span**. It does not prove that subspace orientation is the sole cause. The PCA patch mean norms are 2.5% to 6.2% larger, and the compared models share some examples across training seeds. All assertions about actual trained models are derived from SCM-20's already recorded assay, not from fresh SCM-21 neural inference.

## Related records

- SCM-19 PR #172 and SCM-20 PR #180, now merged into `main`.
- Exact math grader source pinned in SCM-20: EvalMath ARG-01 `c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b`.
- `PROTOCOL.md`: retrospective status, endpoint definitions and caveats.
- `RESULTS.md`: per-seed results, sharp counting theorem and synthetic sanity control.
- `NEXT_TEST.md`: prospectively suggested norm-matched assay for the external 3050 host. The suggested experiment has not been run.
