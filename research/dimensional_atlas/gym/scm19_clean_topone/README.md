# SCM-19: acquire clean mathematics before testing mediation

This registered follow-up changes the public numeric representation and uses
verified set-valued program supervision. It compares two encodings, selects
only by development data, repeats three seeds, and freezes models/dictionaries/
directions before generating fresh clean and semantic-confirmation cases.

See [PROTOCOL.md](PROTOCOL.md) and [config.json](config.json) for the preregistration.
No official EvalMath roster is opened. A clean prediction is one neural argmax,
without candidate verification or search; the independent pinned oracle grades
the selected program afterward. Stronger generic polynomial preprocessing is
explicitly part of the second arm and is not evidence of raw-token reasoning.

The model is FP32, four visits, batch eight, on the local RTX 3050 only. A
3000 MiB allocator cap and 500 MiB sampled headroom requirement are enforced.
The existing Python 3.12/PyTorch 2.9.1+cu126 environment is reused.

From repository root, using a new private directory outside Git:

```powershell
python -m unittest discover -s research/dimensional_atlas/gym/scm19_clean_topone/tests -v
python -m research.dimensional_atlas.gym.scm19_clean_topone.campaign --private ../scm19_private --oracle ../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py --old-private ../../outputs/private_artifacts
```

`results/registration.json` seals configuration and implementation bytes before
training. The local worker uses append-only work journals, optimizer resume
snapshots and immutable completed-stage receipts. New held-out gold and heavy
checkpoints remain in the private directory. Existing SCM-18 artifacts and its
draft PR are preserved. Hosted experiment workflows must be skipped with
`[skip ci]` on publication.
