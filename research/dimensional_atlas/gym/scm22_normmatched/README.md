# SCM-22: CPU norm-matched causal dimensional atlas

Fresh **synthetic trained** four-visit transformer experiment, following [SCM-21](../scm21_capacity/README.md). The protocol was committed as `PRE_REGISTRATION.md` at `56c14ace` **before the held-out experiment**. `PROTOCOL.md` is the frozen extended protocol with registered SHA-256 `5daa2344...`.

## Reproduce

From this directory, using Python 3.13, CPU PyTorch 2.10.0 and NumPy 2.3.5:

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m unittest discover -s tests -v
python run.py --check
python diagnostic.py --check
```

The default experiment trains three new 48-hidden, four-head, four-visit **weight-tied** attention models for 400 AdamW steps each, fits a shared TopK-8 SAE for 250 steps, selects four-dimensional frames from separate calibration pairs, then runs 256 fresh counterfactual pairs per seed. The exact oracle computes two affine maps over the finite field F11. All 768 pairs are clean-correct. We use the same rank-four patch **norm on every pair**, across candidate directions. No RTX 3050 or cloud accelerator is required.

## Primary registered causal experiment

The recipient and donor hold the second mathematical operator, operand and nuisance token fixed, and differ only in the first operator. At zero-based recurrent visit 1, the recipient CLS receives a source-to-donor difference projected into a selected four-dimensional space and rescaled to the full difference's L2 norm. Donor-program correctness is scored independently using exact modular arithmetic.

Registered arms: no-op, full CLS, donor-difference PCA rank four, raw-state PCA rank four, TopK SAE-derived linear-head fiber rank four, and random linear-head fiber rank four. The generated `results/scm22_public_counts.json` records the public condensed receipt; the full per-operator/pairwise reports are available in the accompanying research artifact. `RESULTS.md` contains the final claims and key limits.

## Post-hoc diagnostic

The second script fits rank-four difference PCA **within** the linear head kernel and **within** the output-head rowspace, with a four-visit sweep. It was written after the registered results were inspected. This is exploratory evidence only and does not alter the registered success gate.

The output layer contains LayerNorm, so its linear-head kernel is not globally silent to the full nonlinear readout. This is a controlled geometric instrument, **not** a fitted Goodfire VPD, a trained SCM-20 checkpoint evaluation, or a certified nonabelian braid.

## Source integrity

`model.py`, `geometry.py`, and `run.py` have pinned SHA-256 hashes in the public receipt. The registered experiment uses the independent finite-field oracle and seed-locked model/SAE/calibration/held-out generator roles. Full exact reruns use `--check` against the external complete receipt; hosted CI tests scientific gates and invariants against its own fresh CPU output.

Private trained SCM-20 weights remain unavailable to the sandbox. Transfer to the original model requires separate checkpoint-locked experiments and a clean/OOD accuracy preservation gate.
