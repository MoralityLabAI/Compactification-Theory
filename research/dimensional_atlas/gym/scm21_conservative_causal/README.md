# SCM-21: conservative causal training with a clean-accuracy buffer

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
