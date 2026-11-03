# Algernon H-M0/H-M1: recovered CPU experiment implementation

This folder contains byte-identical imports of the original ChatGPT sandbox's runnable H-M0/H-M1 experimental implementation and the CPU toy contracts. Every imported file has been checked against the archived Git blob hash. See [RECOVERY_STATUS.md](RECOVERY_STATUS.md) for SHA identities, split/seed commands, and exact local setup.

**Now available:** the H-M0 information-gain pruner, terminal-utility controller, cost-aware stopping control, H-M1 bounded grammar revision controller, and all five associated test modules. The optional historical timing benchmark script is still missing.

**Evidence boundary:** these are synthetic CPU experiments. The original raw JSONL records and PyTorch checkpoint binaries remain in the chat sandbox archive and are not yet in this GitHub branch. They can be regenerated from the preserved scripts and seed ranges; independent reproduction should record the new environment and hashes. H-M2 still lacks its original implementation and receipts, so previously reported H-M2 numbers are unverified.

## Run tests

From the Compactification-Theory checkout root, with `numpy` and CPU `torch` installed:

```bash
PYTHONPATH=research/algernon/metacognition python -m unittest discover -s research/algernon/metacognition/experiments/tests -v
```

On PowerShell:

```powershell
$env:PYTHONPATH = "research/algernon/metacognition"
python -m unittest discover -s research/algernon/metacognition/experiments/tests -v
```

The original extracted source suite passes 27/27 unit tests in the sandbox. Rerun the GitHub copy independently before treating its results as a reproduced observation. Training commands are in RECOVERY_STATUS.md.
