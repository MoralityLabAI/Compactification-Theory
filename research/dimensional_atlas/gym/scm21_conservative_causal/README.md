# SCM-21: conservative causal training with a clean-accuracy buffer

Completed: clean accuracy met the two-point guard in all six fresh seed/split comparisons (worst loss 0.49 percentage points), but trained native VPD-write mediation was 0/128 in every seed and stratum. The combined mediation-improvement hypothesis was not confirmed. All six development causal endpoints were also zero; the registered tie-breaker selected lambda .25 at 1000 updates.

| Mean fresh clean accuracy | SCM19 warm | Matched control | Causal |
|:---|---:|---:|---:|
| ID | 66.05% | 66.28% | 66.11% |
| OOD | 43.62% | 43.65% | 43.26% |

See [RESULTS.md](RESULTS.md), [THEORY.md](THEORY.md), and [summary JSON](results/summary.json). Fresh data cover all 16 OOD family/topology combinations, with zero encoded-input overlap against the registered pre-final catalog.

This preregistered follow-up increases clean distillation from 2 to 10, reduces
causal weights to .25/.5 and learning rate to .001, and caps each actual adapter
parameter increment at L2 .01. The original SCM-19 reasoning model, SAE and
head-fiber projector remain frozen. A lambda-zero arm matches parameter and
training-update budgets. See [PROTOCOL.md](PROTOCOL.md) and [config.json](config.json).

Development selection requires <=1 percentage-point clean loss in both splits
and at least 64 common-eligible pairs per stratum. Final confirmation retains
the <=2-point guard. Fresh clean pools contain 1024 ID and 1024 OOD cases, and
fresh pair pools contain 512 ID / 1024 OOD pairs. Every new role is disjoint by
actual encoded input, as well as public signature; OOD covers all 16
family/topology combinations. These distribution changes limit direct comparison
to earlier studies; within-study models use the same fresh cases.

All computation remains local on the RTX 3050 4GB, FP32, full-model batches <=8,
four visits, a 3000 MiB allocation cap and 500 MiB sampled free-headroom guard.
Weights, caches and gold stay outside Git. Publication commits use [skip ci].

```powershell
python -m unittest discover -s research/dimensional_atlas/gym/scm21_conservative_causal/tests -v
python -m research.dimensional_atlas.gym.scm21_conservative_causal.campaign --private ../scm21_private --old-private ../../outputs/scm19/private_artifacts --previous-private ../../outputs/scm20/private_artifacts --oracle ../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py
```

The registration precedes all new training/development scoring. Historical
catalog reconstruction and small CPU fixtures are permitted preflight checks.
Completed receipts cannot be overwritten. A new study requires a new namespace.
Historical source and earlier experiment artifacts remain unchanged.

Eleven unit tests pass. Independent CPU replay regraded 138,240 predictions,
checked 720 mediation cells, all nine frozen models, 56 balanced family/topology
cells, six development choices and 5,120 novel encoder inputs. Every actual
parameter step obeyed the .01 L2 cap; maximum observed was .00999952. Peak
PyTorch allocated/reserved VRAM was 24.49/30.00 MiB, minimum sampled free
3245.7 MiB. Training and model inference used CUDA; generation/oracle checks and
independent verification used CPU. The worker completed without exceptions.

After artifact delivery, replay with:

```powershell
python -m research.dimensional_atlas.gym.scm21_conservative_causal.tools.audit --private ../../outputs/scm21/private_artifacts --old-private ../../outputs/scm19/private_artifacts --previous-private ../../outputs/scm20/private_artifacts --oracle ../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py
python -m research.dimensional_atlas.gym.scm21_conservative_causal.tools.report --private ../../outputs/scm21/private_artifacts
```

Historical model.py/tasks.py Windows runtime CRLF pins differ from LF Git blobs.
The delivered bundle preserves registered runtime bytes with a manifest while
leaving those historical source files unchanged. No hosted experiment or merge
is requested.
