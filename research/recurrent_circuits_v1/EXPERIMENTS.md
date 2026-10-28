# RC-01 experiment protocol: precision, active memory and recurrent execution

Version 1, 2026-10-29. Prospective except for the explicitly recorded CPU reference
checks. No GPU job, model download, workstation process or cloud service is started
by publishing this protocol. Existing E10, IC-01 and loop-capacity runs are unchanged.

## Questions and primary endpoints

**RC-A, mathematical positive control.** Run the exact GF(2) quotient and restored
NAND reference. Primary endpoints: agreement with an independently enumerated
behavioral partition; exact encoding/transition commutation; every tested NAND
output correct inside the stated perturbation box. Failure blocks later claims
about a circuit-preserving compression pipeline.

**RC-B, recurrent numerical reach.** Compare a known circuit with and without
code restoration under a frozen sequence of writes/logic operations. Primary
endpoint: first incorrect logical transition, using an independent symbolic
oracle at every boundary. Report survival at each horizon and all failures.
Compiled operations must be inside the declared tensor forward path. External
host evaluation of the arithmetic is a separate control with its cost recorded.

Start with a parallel NAND ring and affine overwrite/shift machines. Use memory
widths 16, 64, 256; horizons 32, 128, 512, 2048; initial-state seeds 0,1,2 for
calibration and 10..19 for held-out evaluation. Full Cartesian sweeps are deferred
until a small-case smoke test passes and a job budget is selected. Precision arms:
FP32, BF16, FP16, followed by explicitly specified simulated fixed-point formats.
The format, saturation rule, reduction order and rounding mode are experimental
inputs. Int4 storage and an int4 compute kernel are different arms.

Include no-noise, bounded state perturbations, weight rounding, off-target address
mass, and unrestored idle cells. Restore all live cells or specify exactly how
idle errors stay bounded. The restoring map sees only its current scalar input;
it cannot consult the gold symbolic state. Record exact addresses separately from
softmax address lookup. Derive the certified logit gap before choosing attention
scales. A correct answer with an incorrect internal trace fails the trace criterion.

Acceptance for a claimed certified regime: every tested state/operation satisfies
the margin inequalities and every symbolic transition agrees. Finite samples alone
do not establish a uniform theorem for learned weights. A single within-assumption
counterexample falsifies the claimed certificate. Out-of-margin success is useful
empirical evidence but does not extend its theorem. Numerical-device mismatch is
reported separately from an unsupported operation or missing observation.

**RC-C, learned controller and state quotient.** After RC-B, use a tied one- or
two-block controller in roughly the 1-5M trainable-parameter range, selected by
counting the actual module including embeddings/readout. Train on short affine
programs and NAND programs with the same declared instruction interface. Compare
compiled weights, learned weights, learned weights plus the fixed restoration map,
and an untied same-width controller. Keep the compiled arm as an existence and
pipeline control. It does not score as a training-method improvement.

Primary endpoint: held-out exact trace accuracy at 4x the training horizon for
fixed program/working-memory budgets. Separate the fixed-memory horizon sweep
from a fixed-horizon memory-width sweep. Use fresh programs and input values;
never carry held-out programs in an uncounted lookup table. Include random-label
memorization only as a separate description-capacity control.

Run two distinct comparisons: same width and executed block count (compute control,
untied model has more weights), and parameter-budget matching (width/state must
then be reported explicitly). Training tokens, optimizer updates and training FLOPs
must all be logged. The existing loop-capacity owner's untied control is read-only;
this protocol does not change its frozen sweep or claims.

For state compression, first use the affine answer key: parity under permutations
requires one bit; enabling arbitrary writes requires all m bits. Compare probe rank,
linear/GF(2) recovery where appropriate, and learned encoders at declared bit counts.
A state collision with different legal future outputs falsifies sufficiency.
Real-valued embedding rotations require consistent transformation of reads, writes
and decoder; rotating the state alone is an explicit negative control.

## RC-D: GPU/host-DRAM residence experiment

Use the RTX 5080 16 GB as the first planned device. Its currently free memory and
host RAM capacity are unverified. Inspect them immediately before any run. The
workstation is not reachable from this execution environment. Use one bounded job
at a time, with no assumption that other owners' allocations are free.

The delivered `gpu_dram_bench.py` contains two pilots:

- `nand`: a fixed repeated tensor NAND circuit, with independent Boolean replay.
- `transformer`: an untrained full-sequence transformer block with shared weights,
  repeated on the same state. This measures systems behavior only.

Compare resident execution with explicit lossless CPU roundtrips every k loops.
The script records source hashes, dtype, tensor shape, parameter bytes, copy bytes,
wall times, output agreement and GPU allocator peaks. CPU-only smoke runs are
allowed. CUDA requires the explicit `--confirm-gpu` flag. The current script is
inference-only, uses one CUDA stream, and makes no overlap claim.

Start at T=128, k in {1,8,32}, batch=8, tokens=512, width=256 in BF16. The recurrent
state is 2 MiB. Its defined schedule moves 512 MiB at k=1 and 16 MiB at k=32, including
the final roundtrip. These are byte counts, not measured speedups. Establish the
resident reference and trace/transport equality before timing comparisons. Use
FP32 first where required to diagnose precision changes.

This first pilot retains a full device state and therefore establishes no VRAM
capacity gain. A separate multitenant extension must keep inactive session states
only in host DRAM and rotate an active microbatch through GPU memory. Compare equal
logical workloads and show actual allocator residency, including a clean oversized
resident baseline that reports OOM rather than silently reducing batch size.

Proposed multitenant sweep: hold total stored sessions fixed, vary active sessions
and loops per residence. Record per-session latency, useful transitions/second,
correct completed tasks, peak allocated/reserved VRAM, host RSS, pinned host bytes,
H2D/D2H traffic and initialization costs. A large host-memory pool is only useful
when the live working set is small enough or transfers can be amortized.

Random pointer chasing is an adversarial locality control. A next address depending
on the just-fetched word prevents arbitrary prefetch. Dense sequence-state attention
has a different I/O graph; paging one state tile does not preserve its computation
unless attention and updates use a separately verified streaming algorithm.

A small weight-tied block fitting in VRAM is the initial architecture. Profile L2
hits and memory throughput before claiming on-chip weight reuse. Use measured
H2D/D2H bandwidth and synchronization latency in the residence-time inequality in
THEORY.md. Include model-loading and terminal-copy costs in a separate end-to-end
measurement, since the delivered pilot isolates the hot recurrent region.

Acceptance for a practical DRAM claim: preserved outputs/traces at the declared
precision, equal workload, higher feasible concurrency or throughput per declared
VRAM budget, and complete traffic/memory records. A time reduction with a smaller
workload is invalid. A correct slowdown records the tradeoff. Missing GPU telemetry
leaves the performance claim unmeasured.

## KV-cache and training extensions

For a decoder with per-loop caches, compare full independent caches, shared first-loop
cache plus local windows, and a loop-indexed compressed cache. Report total physical
cache bytes, recurrent-state bytes and executed work. The first variant has an RN
factor; sharing changes it to N+(R-1)*min(w,N). Preserve the original decoder or train
for its new memory semantics; a post-hoc collapse has its own quality gate.

Weight sharing reduces unique parameters. Full backpropagation through T iterations
still retains or recomputes a T-dependent activation graph. Activation checkpointing
adds work; truncated backpropagation changes the gradient. Record those choices
instead of assuming inference's state budget also holds during training.

Primary sources for implementation details:
https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
https://docs.pytorch.org/docs/stable/notes/cuda.html
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

Preallocate pinned buffers outside timing. Synchronize before a CPU reads an
asynchronously copied result. The initial pilot deliberately serializes all
transfers and compute; double buffering and independent-stream overlap are
prospective experiments with separate lifetime/synchronization tests.
