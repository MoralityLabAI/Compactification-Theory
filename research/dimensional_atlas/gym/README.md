# SCM-15 gym: SAE + Atlas-Transport VPD + exact math feedback

The gym starts with an executable tiny looped transformer; it is designed to attach to the pinned independent EvalMath formal-series oracle and later to RMP pretrained checkpoints. Its initial finite-ring math backend is **synthetic**, with distinct algebra from EvalMath. All mathematical judgments remain exact, and no candidate program is graded on future coefficients until candidate selection is frozen.

## Atlas-Transport VPD

An identical physical attention+MLP block runs for every recurrent visit. A shared TopK SAE fits visit-spanning residual features. The AT-VPD adapter applies a shared rank-`r` low-rank correction whose coefficients are the product of a *sparse feature router* and a *visit gate*. Base model and SAE are frozen for adapter tuning. This is a new, exploratory visit-conditioned residual/parameter-fiber adapter inspired by VPD, rather than a reimplementation of Goodfire's original decomposition algorithm. The fitted adapter makes the effective transformation visit-dependent; the pure tied baseline and shuffled-visit-gate control are always reported.

Training sequence: fit strictly tied baseline on mathematically specified programs; fit SAE from frozen residual traces; train AT-VPD while freezing the baseline and SAE; mine high-confidence exact-prefix verifier failures on the discovery set; synthesize new replay tasks in the selected family/topology/program cells; continue adapter training; freeze all parameters and task-selection decisions; score final test and control arms. The new generated *evaluation* challenges use separate source seeds and gold files, never recycled as training tasks.

### Run locally (synthetic math)

From `Compactification-Theory` repository root with PyTorch installed:

```sh
python -m unittest discover -s research/dimensional_atlas/gym/tests -v
python -m research.dimensional_atlas.gym.run \
  --backend toy --steps 40 --sae-steps 25 --vpd-steps 45 --replay-steps 20 \
  --output /tmp/scm15_toy.json
```

Three outputs are created, and existing paths cause refusal: `scm15_toy.json` (source/metrics/manifest hashes), `scm15_toy.generated.public.json` (public generated tasks), and `scm15_toy.gold.private.json` (separately stored private labels, not fed to training). Keep the private labels outside published model-facing bundles.

### Independent EvalMath ARG-01 mode

The independent oracle source belongs to `MoralityLabAI/EvalMath`, with the frozen revision and SHA-256 in `research/algernon/algebraic_reasoning_gym/evalmath_pin.json`. A local checkout at that revision is required; the loader compares source bytes against the pinned SHA-256 and refuses drift.

```sh
python -m research.dimensional_atlas.gym.run \
  --backend evalmath \
  --evalmath-source ../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py \
  --output /tmp/scm15_arg01.json
```

Training and discovery only generate fresh `signed`, `product`, `sparse` families. Final readout uses the official already-public EvalMath roster, with its OOD families, negatives and finite-prefix collision pairs, first accessed *after* training and task mining are frozen. This is an in-repository research evaluation, not a new blinded contest. Prefix checking is independent of the model. Pass rates, false-future acceptances, mean paid verifier calls and family breakdown remain separate.

### Frontier / local-maximum criterion

For rejected top candidate `g` on discovery item `x`, with confidence `p` and first mismatching observed coefficient `f`, define `proxy(x)=p*(f+1)/13`. Within each `(family, topology, source_program)` cell, retain the highest proxy under a deterministic tie break. Those are local peaks of a **finite discovery proxy**, not proven local maxima of a neural loss or the original EvalMath domain. Generate new training examples and a separate, uninspected evaluation set from the selected cells with fresh random seeds and public-signature deduplication. Disallow mining from held-out sets.

### Mechanistic dimension measurements

`model.atlas_probe` estimates two tangent singular values, subspace alignment across visits and a second-difference norm in a chart fixed by two SAE decoder directions. These are numerical diagnostics, with finite-angle truncation and projection dependence. SCM-05/13/14 contain the separate source certificates needed for stronger topological claims. Subsequent work can add exact math intermediate-variable intervention, head-orthogonal fiber constraints, matched FLOPs, confidence intervals over independent source families, and RMP 64-token/128-hidden C3 checkpoint hooks.

**Current scope:** the local toy run can show that the training and evaluation machinery works. It makes no claim that AT-VPD improves generalization or that the new generated tasks are independently blinded. The EvalMath adapter requires the separately pinned oracle checkout, which is not bundled here. RMP weights and D-shared dictionaries remain external.
