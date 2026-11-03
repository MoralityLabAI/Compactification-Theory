# Recovered Algernon CPU experiment files

The first two executable files were recovered from the chat sandbox ZIP and pushed **byte-for-byte** to this GitHub branch.

| File | Verified Git blob SHA |
| --- | --- |
| `experiments/toy_contracts.py` | `16a92153570e943c4b2e4287bcd7f8ab66c309bb` |
| `experiments/tests/test_toy_contracts.py` | `21805955f4ab283cc0501d608081b177fba57192` |

These files implement the small software-only RV32 subset and a reversible recurrent-graft gate; this is *not* Qwenputer's complete RV32 circuit. The original archive contains additional experiments (H-M0 and H-M1), reports, checkpoints, and raw records which have **not yet been transferred** to GitHub. H-M2 has no retained reproducible artifacts.

## Run on a checkout

From the Compactification-Theory repo root (requires `numpy` and CPU `torch`):

```bash
PYTHONPATH=research/algernon/metacognition python -m unittest discover -s research/algernon/metacognition/experiments/tests -p test_toy_contracts.py -v
PYTHONPATH=research/algernon/metacognition python research/algernon/metacognition/experiments/toy_contracts.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "research/algernon/metacognition"
python -m unittest discover -s research/algernon/metacognition/experiments/tests -p test_toy_contracts.py -v
```

## Recovery status

The only verified complete sandbox ZIP is still at the conversation's container path `/mnt/data/algernon_claude_handoff.zip`, SHA256 `88d8e23581f0445aca9ac7b7b9a9c7f8d7df34f00bbc5126fced84511053f7b4`. The linked private Hugging Face dataset is a proposed destination; no round-trip upload was confirmed.

Next: recover the remaining 9 Python files, their manifests, checkpoints, records, and logs, preserving original hash identities.
