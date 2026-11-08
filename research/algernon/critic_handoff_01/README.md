# TRSI-CRITIC-01: critic handoff under policy and reward drift

Prepared 2026-11-09 for Compactification-Theory's Algernon track.

Status: exact finite-state arithmetic executed. Neural critic transfer and recurrent-graft training remain unrun.

This adds a critic-calibration design alongside `research/algernon/` and its existing algebraic-machine work. It does not replace the existing quotient/oracle work or rerun archived checkpoint recovery. Preserve the external-baseline pivot in `research/algernon/README.md` and all earlier negative results.

```bash
cd research/algernon/critic_handoff_01
python -m unittest discover -s . -p 'test_*.py' -v
python run_cpu.py
```

`critic.py` uses rational Gaussian elimination for small exact examples. Bellman sweeps are numerical reference operations, not SGD updates or a reproduction of the paper's four-update calibration.
