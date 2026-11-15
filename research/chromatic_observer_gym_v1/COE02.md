## COE-02: exact ℓ1 cut-capacity and quantization gap

**November 15, 2026.** This extends COE-01 on the existing draft research branch.

Each binary probe is an oriented cut of a finite sample of latent states. Its
pairwise Hamming contribution is a cut semimetric. The exact minimum binary
width under an `ell_1` Lipschitz constraint is therefore a nonnegative integer
cut-multiplicity covering problem with per-pair capacity constraints. The
companion [MathCan theorem and proof](https://github.com/patrickdugan/MathCan/blob/feat/coe01-chromatic-observer-20261115/research/compactification/CHROMATIC_CUT_CAPACITY.md)
explains the exactness, rational dual lower certificates and the sharp
unbounded gap between fractional cut-mass and physical binary coordinates.

### Reproduce

Run from this repository's root with Python 3.10+ and its standard library:

```bash
python -m unittest discover -s research/chromatic_observer_gym_v1 -p 'test_*.py' -v
python research/chromatic_observer_gym_v1/l1_cut_codes.py
python research/chromatic_observer_gym_v1/activation_case.py research/chromatic_observer_gym_v1/frozen_k5_onehot.json
```

The frozen fixture is a **synthetic**, rational five-state, five-feature
one-hot activation prototype with equilateral `ell_1` metric. It requires all
pairs distinguished. These are mathematical test states, not model activations.
The exact solver and a separate word-level exhaustive oracle agree across all
8 three-state evaluator graphs at `L=0,1,2` with Hamming margins 1 and 2.

| Equilateral K5 evaluator | Minimum bits | Method |
| --- | ---: | --- |
| `ell_infinity`, Lipschitz `L=1` | 3 | COE-01 graph coloring |
| `ell_1`, `L=1` | infeasible | odd-cycle/cube-diameter obstruction |
| `ell_1`, `L=2` | 4 | exact cut BFS, sharp anticode theorem |
| `ell_1`, `L=3` | 3 | exact cut BFS and witness |
| fractional cut LP, `L=2` | 5/3 | balanced rational cut witness and dual |

The integer/fractional ratio is `12/5` for K5 and grows on the order of `n/2`
for equilateral complete evaluators K_n. Fractional cut mass is only a
mathematical relaxation: its value does not measure the number of discrete
feature coordinates a network has learned.

### Real activation adapter

`activation_case.py` accepts a frozen JSON document with `format` set to
`coe02-activation-v1`, a list of states with unique `id` and equal-length
`activation` feature vectors (rational or decimal strings), `evaluator_edges`
of state-ID pairs, `lipschitz` (rational string), and `hamming_margin`.
The adapter computes the *exact rational feature-ell_1 metric* and compares the
COE-01 `ell_infinity` observer with the COE-02 `ell_1` optimum, preserving an
explicit provenance string. For larger datasets, prototype selection and the
evaluator split must be frozen before choosing probes. The exact method is
small-n exponential and raises when a resource guard is hit.

Current evidence is synthetic and combinatorial. Model checkpoint extraction,
trained/held-out SAE probes, intervention response, and empirical mechanism
claims remain future verification gates.
