# RRC-04: learning complete partition programs from raw coefficients

**Executed 2026-11-09, CPU-only.** Three trainable toy networks, 1,500
training examples, exact source-family holdouts, verified program checker,
and a finite-code baseline. The result is primarily a **negative neural
transfer result** with a strong exact symbolic comparator.

## What the student received

Input is 65 exact integer counts, degrees `0..64`, transformed only by
`log1p`, first neighboring log difference, and public degree index. The
neural student predicts a multiplicity cap, an arithmetic period and the
complete periodic residue mask. The shared exact checker independently
executes that program through degree 64, then grades its degrees 65–128.
Training used only periodic-product sources; local gap constraints and
nonperiodic arithmetic part sets were withheld as source families. Product
moduli 5, 7 and 11 were withheld as a separate modulus-disjoint test.

The experiment also includes a *hybrid* mode in which a neural cap/period
proposal is completed with a mask by exact Euler inversion. That extra
symbolic calculation is accounted for independently of direct neural
program output.

## Measurements

| Predictor, depth | Verified complete programs /351 | Correct future /351 | Hybrid verified /351 | Inference MACs/case* |
|---|---:|---:|---:|---:|
| Exact symbolic grammar synthesizer | 307 | 299 | n/a | integer exact solver |
| GRU student, 0 extra loops | 0 | 0 | 32 | 127,008 |
| GRU student, **2 trained loops** | **0** | **0** | **28** | **128,160** |
| GRU student, 8 extra loops | 0 | 0 | 17 | 131,616 |
| Full-history feedforward, 2 layers | **6** | **6** | **69** | **3,297** |
| Shuffled-label GRU, 2 loops | 0 | 0 | 17 | 128,160 |

\* Multiply–accumulate estimates exclude the nonlinear gate operations,
model training, exact series computation, checker, and Python runtime.
There are 180 development examples and **171 test examples** in the
351-case denominator. No model selection was performed on the test data.
All six directly accepted feedforward programs occur on development cases;
all 171 test programs are rejected by every neural direct proposer.

### Exact baseline by test family

| Family | Prefix-executable certificates | Correct all degrees 65–128 | Expected interpretation |
|---|---:|---:|---|
| Development periodic products (180) | 180 | 180 | in-class |
| Unseen moduli 5, 7, 11 (108) | 108 | 108 | exact class-complete search |
| Local gap constraints (32) | 3 | 3 | gap 1 minimum 1; both RR gap-two minima |
| Nonperiodic part sets (15) | 0 | 0 | complete within declared bounded class |
| Paired late-defect controls (16) | 16 | 8 | eight future mutations invisible in prefix |

The three positive local-gap cases are **gap 1, minimum 1** (distinct
parts), plus the two known Rogers–Ramanujan identities (gap 2, minima
1 and 2). This is held-out *source-family* transfer for the exact
compiler. The model successfully predicts no unseen local-gap program.

Among the 171 test cases, the exact solver accepts 127 and predicts
119 correctly through degree 128. The eight errors are paired mutants
whose first new part appears after degree 64. Every pair has exactly
the same observable input. Any prefix-only forced full-vector prediction
can be correct on at most one member of each pair. The exact solver
attains that information ceiling by choosing the unmodified base rule.

The neural cap/period plus exact-mask hybrid also struggles on the test
split: the feedforward control recovers five unseen-modulus cases and
one gap case; the recurrent model at two loops recovers zero. Hybrid
failures should not be attributed wholly to the neural model because the
symbolic mask completion and direct finite verification are additional
computational resources.

### Recurrence, resource and caveats

The GRU has **3,363** FP32 parameters (107,616 bare weight bits) and
24 FP32 recurrent state values (768 live bits). The feedforward model
has **3,345** FP32 parameters (107,040 weight bits), with complete
access to all 65 input coefficients. At the trained depth the GRU costs
about **39x** the counted dense multiply–accumulates of the feedforward
control. The 2-loop GRU shows no measured direct-program gain over its
zero-loop setting or the feedforward control, and its exact-mask hybrid
coverage decreases with extra loops. This is one constrained toy pilot,
not a general impossibility result for neural algebra discovery.

The small code lengths in the exact program (a two-bit multiplicity tag,
Elias-gamma period and one bit per residue) cover syntax only. The model
checkpoint, input coefficients, shared compiler, train labels, hidden
memory and verifier are counted separately. The exact solver is a
stronger resource-aware baseline than either neural architecture on
this roster.

Three incidental evaluation prefixes match training prefixes. The
source-bound receipt contains **one local-gap alias and two collision aliases**; these are
reported, with their source provenance, and excluded from any claim of
strict raw-input disjointness. Product development and unseen-modulus
prefixes are deduplicated against training.

## Reproduce

From the repository root, Python 3.10+ standard library:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_program04.py' -v
python research/rrc_compactification/bench04.py --check
```

Optional checkpoint reproduction (PyTorch CPU installed):

```bash
python research/rrc_compactification/train04.py --steps 650 --batch-size 128
python research/rrc_compactification/bench04.py --check
```

The first command writes three checkpoints with input and parameter
hashes. The second checks the full source-hashed replay; any changed
weights or code require explicit receipt regeneration and review.
This reference run used PyTorch 2.10.0+cpu, one CPU thread; training
consumed approximately 22 seconds wall clock in its local container.
The standard-library evaluation takes approximately 11 seconds there.
These are local measurements rather than cross-machine speed estimates.

## Next experiment: RRC-05

Train a recurrent **algorithmic** student to produce intermediate Euler
exponents or formal-series residuals and to carry a verifiable program
trace. Freeze an independent new test roster before any architectural
changes. Test candidate ranking and abstention under matched compute
budgets, and require direct output improvements beyond the exact solver's
class-conditional success. RRC-04's exposed 171-case test roster must
be treated as development data in any subsequent iteration.
