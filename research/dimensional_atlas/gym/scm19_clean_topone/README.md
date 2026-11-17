# SCM-19: acquire clean mathematics before testing mediation

This registered follow-up changes the public numeric representation and uses
verified set-valued program supervision. It compares two encodings, selects
only by development data, repeats three seeds, and freezes models/dictionaries/
directions before generating fresh clean and semantic-confirmation cases.

Completed: mean clean top-one accuracy improved from 4.1% to 60.9% ID and
4.9% to 44.1% OOD, using the simpler coefficient encoding. With the frozen-base
SAE/VPD adapter, accuracy was 62.6% ID and 45.3% OOD. The primary native-write
mediation endpoint was 0/64 ID in all three seeds, matching random controls;
OOD was also zero, with two seeds below the registered eligibility minimum.
The clean acquisition result is positive; the registered semantic mediation
claim is unsupported. See [RESULTS.md](RESULTS.md), [THEORY.md](THEORY.md), and
[results/summary.json](results/summary.json).

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

The recorded campaign's first mediation logging call raised a payload-name
collision. [RECOVERY.md](RECOVERY.md) describes the separately committed,
logging-only recovery; original preregistered source bytes remain intact.
The actual resumed entry point was `tools.resume` with the same arguments.
The completed receipts intentionally prevent retraining or overwriting a
recorded evaluation. A new acquisition study needs its own registration and
results namespace.

Audit the completed local artifacts and regenerate the figures with:

```powershell
python -m research.dimensional_atlas.gym.scm19_clean_topone.tools.audit --private ../../outputs/scm19/private_artifacts --oracle ../EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py
python -m research.dimensional_atlas.gym.scm19_clean_topone.tools.report --private ../../outputs/scm19/private_artifacts
```

All eight unit tests pass. An independent replay verified 28,672 predictions,
480 mediation cells, freeze chronology and cumulative model/evaluator counts.
Peak allocated/reserved VRAM was 40.90/52.00 MiB, with 3223.7 MiB minimum sampled
free memory. Interrupted-process post-data pure-oracle/cache counters are
explicitly incomplete; cumulative check/grade and model counts are exact.

Byte pins cover the Windows runtime checkout. Historical `model.py` and
`tasks.py` have CRLF runtime bytes and LF Git blobs. They remain unchanged.
The delivered research bundle preserves their exact registered runtime bytes
and records both hashes, so unpacking that bundle reproduces strict source-pin
checks without changing historical repository files.

`results/registration.json` seals configuration and implementation bytes before
training. The local worker uses append-only work journals, optimizer resume
snapshots and immutable completed-stage receipts. New held-out gold and heavy
checkpoints remain in the private directory. Existing SCM-18 artifacts and its
draft PR are preserved. Hosted experiment workflows must be skipped with
`[skip ci]` on publication.
