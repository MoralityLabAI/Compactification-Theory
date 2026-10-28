# Recurrent circuits and resource-bounded compactification, RC-01

Additive research package, 2026-10-29. Baseline inspected:
`MoralityLabAI/Compactification-Theory@fc27a7d33aa483411219b8da3991254d1fd174d3`.

`THEORY.md` derives a resource ledger for embedding finite circuits in repeated
neural blocks. The main new worked result is an exact minimal-state quotient for
affine binary circuits under a declared intervention alphabet. The note also gives
a uniform restoration condition for recurrent logical accuracy and connects the
precision condition to the repository's weight-sensitivity account. All proofs are
handwritten. Gaussian closure is checked against independent finite-state enumeration.

`EXPERIMENTS.md` specifies numerical reach, learned-controller and GPU/host-memory
experiments. Existing QPC-T01, IC-01, EvalLang, E10 and loop-capacity work is untouched.
No existing model, runtime policy, CI workflow or owner branch is edited.

## Run the exact checks

Python 3.10 or later; standard library only:

```bash
cd research/recurrent_circuits_v1
python -m unittest discover -v
python -O -m unittest discover -v
python verify.py --check results/reference_v1.json
python -O verify.py --check results/reference_v1.json
```

The suite has 23 test methods. It checks 160 seeded affine systems, quotient
transitions and minimality against an independent partition oracle, the parity
write-intervention obstruction, finite-description counts and robust NAND logic.
The recurrent NAND test covers all 16 initial four-bit states for 128 steps with
bounded injected errors. Validation also rejects accidental GPU activation and
CPU copies masquerading as GPU/DRAM measurements.

## Systems pilot

PyTorch is optional and is used only by `gpu_dram_bench.py`. The supplied execution
environment has PyTorch 2.10.0+cpu. Both CPU smoke commands were executed; CUDA was
unavailable. Their raw observations and logs are in the downloadable source bundle.
Those smoke observations establish code-path execution, with no GPU speed claim.

```bash
python gpu_dram_bench.py --kernel nand --loops 128 --batch 2 --tokens 16 --width 16 --repeats 2
python gpu_dram_bench.py --kernel transformer --loops 4 --batch 1 --tokens 8 --width 16 --repeats 2
```

A future GPU run requires a deliberate flag and measured free memory. Example
commands, after installing a compatible local PyTorch build and completing the
workstation preflight in EXPERIMENTS.md:

```bash
python gpu_dram_bench.py --device cuda --confirm-gpu --kernel transformer --dtype bfloat16 --loops 128 --batch 8 --tokens 512 --width 256 --mode resident --out results/gpu_resident.json
python gpu_dram_bench.py --device cuda --confirm-gpu --kernel transformer --dtype bfloat16 --loops 128 --batch 8 --tokens 512 --width 256 --mode spill --chunk 32 --out results/gpu_spill_k32.json
```

The transformer pilot uses randomly initialized shared weights, a full sequence
state and no autoregressive KV cache. It benchmarks transport and repeated block
execution. Its spill arm preserves the device state allocation, so it measures
roundtrip overhead without claiming VRAM capacity savings. Real DRAM-backed session
paging and trained-loop cache sharing are separate prospective experiments.

Each benchmark records wall time, tensor/parameter sizes, copy bytes, output hashes,
source hashes and available allocator counters. The hot-region timing excludes load
and terminal-copy costs. The time cap is cooperative and allocator estimates reserve
headroom; neither provides a process-level hard memory or execution watchdog.

## Evidence status

Exact finite checks and CPU smoke runs have executed. Learned-controller training,
GPU/PCIe measurements, Qwenputer weight loading, full-repository CI and Lean kernel
verification remain unperformed. The shared-memory literature and compiler papers
are attributed in THEORY.md with their own assumptions. Claims about learned
models discovering the constructed mechanisms require their own experiments.
