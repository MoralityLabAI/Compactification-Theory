# SCM-18: RTX 3050 mechanistic-interpretability gym

Local FP32 experiments extend SCM-15/16 with a device-aware 139K-parameter
transformer, an exact head-fiber adapter, three architecture seeds, four
training objectives, discovery-only mathematical local maxima, and a separate
freeze-before-final process. All models are trained from scratch. See
[RESULTS.md](RESULTS.md) for executed results and limitations,
[PROTOCOL.md](PROTOCOL.md) for the registration, and [THEORY.md](THEORY.md)
for the distinction between linear fibers and local nonlinear margin fibers.

## Reproduce locally

Use Python 3.12, PyTorch `2.9.1+cu126`, NumPy and Matplotlib. No Transformers,
cloud hardware, downloaded model weights, or hosted training are needed.
The exact installed package versions and repository revisions are recorded in
`results/provenance.json`. Check disk space before installing CUDA wheels.
Install the pinned GPU build using the official CUDA 12.6 wheel index; the
remaining dependencies use PyPI. The requirements file records the exact build.

From the Compactification-Theory repository root, with a separately pinned
EvalMath checkout at `../EvalMath`, run the following stages **sequentially**.
Use a new output/private directory for a new campaign; existing receipts are
never overwritten. Private checkpoint and gold directories must remain outside
the repository. Replace `python` with the chosen environment's executable.

```powershell
$module = 'research.dimensional_atlas.gym.scm18_rtx3050.campaign'
$oracle = '../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py'
$out = '../scm18_new_results'
$private = '../scm18_new_private'
python -m $module --stage parity --device cpu --oracle $oracle --output $out --private $private
python -m $module --stage parity --device cuda --oracle $oracle --output $out --private $private
python -m $module --stage train --device cuda --oracle $oracle --output $out --private $private
python -m $module --stage diagnostics --device cuda --oracle $oracle --output $out --private $private --certificates ../certificates
python -m $module --stage final --device cuda --oracle $oracle --output $out --private $private
```

After a run, audit the receipts and checkpoint files with
`python -m research.dimensional_atlas.gym.scm18_rtx3050.tools.verify --root research/dimensional_atlas/gym/scm18_rtx3050 --private ../campaign_private`.
For a fresh checkout of these published receipts, historical imported modules
may have LF rather than this Windows run's CRLF bytes. `results/source_encoding.json`
records both domains. The verifier's optional `--allow-checkout-line-endings`
accepts only that text conversion and explicitly reports that the exact
original byte encoding is unavailable. Primary local verification uses strict
byte matching. New SCM-18 files have a local LF `.gitattributes` rule.

For the topology diagnostics, export the unmodified `continuous.py`,
`scm12_rank_one.py`, and `scm13_affine_disk.py` from
`research/seifert_activation/` at certificate commit
`731766c9961314b80c039b77609f3494004124ab` into `../certificates`.
These are separate feature-branch sources, not missing dependencies silently
replaced by new implementations. Their hashes are recorded in provenance.
The diagnostics compare conditional surrogates using sampled errors; they
do **not** establish the routines' continuum premises for this neural model.

```powershell
python -m unittest discover -s research/dimensional_atlas/gym/tests -v
python -m unittest discover -s research/dimensional_atlas/gym/scm16/tests -v
python -m unittest discover -s research/dimensional_atlas/gym/scm17/tests -v
python -m unittest discover -s research/dimensional_atlas/gym/scm18_rtx3050/tests -v
```

If the host's temporary directory is restricted, set `TEMP` and `TMP` to a
writable workspace directory before running tests. No original SCM-15–17 code
or historical receipt is modified. This is a stacked research branch based on
SCM-17; its draft PR targets the SCM-17 feature branch.

## What is measured

`registration.json` fixes pools and discovery choices before fitting.
`training.json` records all development decisions and losses. `freeze.json`
binds model, dictionary, calibration-frame and source hashes before the final
process can open EvalMath's public roster. `final.json` records paired exact
future judgments, paid verifier queries and actual oracle calls.
`diagnostics.json` records two/four-dimensional Jacobians, effective rank,
principal angles, SAE support overlap, curvature, fiber transport, compression,
and a named first-operator interchange test. `rmp_preflight.json` records the
genuine RMP artifact gate separately. New gold sidecars are represented only
by hashes in public receipts.

`rmp_metadata_preflight.json` refines the initial preflight after tracked metadata
were retrieved: only the actual checkpoint/integrity receipt, selected dictionary
and monitor bundle remain absent from that checkout. The additional
`interchange_methods.json` audit covers every intervention family; it was added
after development without changing any frozen model or selection decision.
`discovery_future_audit.json` grades the discovery candidates' withheld futures
only after the main freeze. `original_evalmath_cpu.json` and
`original_evalmath_cuda.json` instrument the complete original SCM-15 runner,
including its discovery/replay and official public readout, with device-only
shims. These supplemental reproductions run after the main freeze.
`native_mediation.json` adds a clearly labeled post-final exploratory check of
all three frozen seeds using their native head, fiber and trained adapter write
directions. It tests the same sixteen pairs, with no new direction/model search.

The first long final process was interrupted before its combined receipt.
`tools/recover_final.py` reuses the original evaluator and unchanged freeze,
persists completed policy/dataset scores, and memoizes only deterministic exact
integer oracle execution. Logical requests and physical cache hits/misses are
separate. `interrupted_final_attempt.json` reports the irrecoverable accounting
gap of that failed attempt. No models or decisions were retuned during recovery.

All inference ablations reuse the same trained factors and frozen base. The
separately fiber-trained and random-dictionary controls have matching factor
shapes and optimizer budgets. An untied comparator starts with cloned core
weights and receives identical steps/examples and expanded visits; its parameter
count is larger. Same arithmetic paths do not guarantee equal GPU timing.

The 12-query mathematical verifier can compensate for a weak neural ranking.
Top-one exact correctness, false future acceptance, and paid queries are
therefore reported separately. A score alone does not show correct internal
reasoning. A zero denominator in the valid counterfactual test yields null
causal fidelity. The SAE reconstruction sweep counts retained features, not
necessarily an equally sized linear state representation. Fiber-preservation
curves retain the head-readable subspace in addition to their displayed fiber
dimension. PCA directions beyond calibration rank are arbitrary orthogonal
completions and carry no estimated-variance significance.
