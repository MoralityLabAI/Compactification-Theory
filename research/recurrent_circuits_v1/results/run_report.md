# RC-01 recurrent-circuit numerical and hardware run report

Pinned source: `8a79b9668d2fc4f2e2028b8914379aded33285f7` (current upstream `main` matched at fetch). Local branch: `research/rc01-5080-experiments`.

## Preflight and gate

- Both required unittest commands passed all 23 tests.
- `python verify.py --check results/reference_v1.json` passed after this Windows checkout was restored to exact Git blob bytes. Git's initial `core.autocrlf=true` had changed bytes that `verify.py` hashes. Local clone config is `core.autocrlf=false`; no pinned source file was edited.
- Host snapshot: 31.12 GiB RAM, 3.15 GiB available (89% load). RTX 5080: 16,303 MiB total; 4,192 MiB used and 58% utilization at the last sample, 68°C and 130 W. Earlier samples showed 40–43% utilization and 2,656 MiB used. WDDM exposed no process owner. Python 3.12.10 had no PyTorch; CUDA/PyTorch compatibility could not be checked. Full snapshot: `preflight_hardware.json`.
- No GPU/DRAM jobs were started. Rising utilization and allocation plus a hidden owner meant the current allocation could not safely be presumed free. CPU-only exhaustive attempts were stopped when scalar emulation took longer than a bounded calibration slot; the completed matrix below is the reduced pre-registered calibration: ring widths 4 and 16, all 16 initial states at width 4, seeds 0,1,2,10,11 at width 16, plus idle widths 16,64,256.

## Completed CPU numerical experiment

This is scalar software emulation with FP32/BF16/FP16 rounding after each modeled primitive, not a CUDA/PyTorch kernel result. Every intermediate logical state was compared against an independent integer NAND-ring oracle. The coherent bounded-error arm adds +1/8 to every lane after every update. The idle-memory arm activates one quarter of the cells per transition and adds +1/8 to every cell; the positive control restores idle cells before the next state use.

| Precision emulation | Restored NAND, no noise | Restored NAND, +1/8 each transition | Gate restoration disabled, +1/8 |
|---|---:|---:|---:|
| float32 | 32/128/512/2048 pass, widths 4+16 | 32/128/512/2048 pass, widths 4+16 | first error at transition 5 (widths 4+16) |
| bfloat16 | 32/128/512/2048 pass, widths 4+16 | 32/128/512/2048 pass, widths 4+16 | first error at transition 5 (widths 4+16) |
| float16 | 32/128/512/2048 pass, widths 4+16 | 32/128/512/2048 pass, widths 4+16 | first error at transition 5 (widths 4+16) |


Restoring NAND passed all 16 width-4 initial states and all five width-16 seeds through 2048 transitions in every emulated precision. With the same bounded perturbation but no gate restoration, the earliest mismatch was transition 5: lane 3, width 4 (all three precisions); width 16 first failed at lane 11. This shows a restoration-dependent stable regime for this constructed circuit and perturbation sequence, not for learned weights or actual GPU arithmetic.

| Idle width | Restoring idle cells | No idle restoration |
|---:|---:|---:|
| 16 | all three precision emulations correct through 2048 | first error at transition 4, idle lane 4 (all three precisions) |
| 64 | all three precision emulations correct through 2048 | first error at transition 4, idle lane 20 (all three precisions) |
| 256 | all three precision emulations correct through 2048 | first error at transition 4, idle lane 8 (all three precisions) |


The observed outcomes agree with the conditional restoration-margin argument in `THEORY.md`, Theorem 5: for the encoded NAND case, separation is 1, radius is 1/8, the restored symbolic map has rho=0 and per-step perturbation eta=1/8; hence 2r<Delta and eta<=(1-rho)r. The theorem supplies arbitrary-horizon correctness only while those uniform assumptions hold. These finite CPU emulations neither establish the assumptions for a GPU kernel nor extend the guarantee to other circuits.

## GPU/DRAM status and conclusions

| Requested question | Evidence in this run |
|---|---|
| Fixed weights and symbolic correctness at extended horizons? | Yes for the explicitly constructed CPU-emulated NAND circuit through 2048 steps; no GPU kernel or learned controller evidence. |
| Numerical restoration stable? | Yes for tested circuit, perturbation, and scalar precision emulations; disabling restoration or idle refresh produced failures. This is consistency with a conditional theorem, not a general empirical horizon for neural circuits. |
| Host DRAM increases feasible concurrent GPU state capacity? | Not measured. No separate multitenant prototype or VRAM residency run was started. |
| Do transfers erase short-residence gains? | Not measured. No H2D/D2H timing, traffic, pinned-memory, or GPU allocator counters are available. |

The supplied CUDA NAND and transformer commands, including the exact batch=8, tokens=512, width=256, loops=128, spill intervals 1/8/32, five repetitions, dtype comparisons and `--confirm-gpu`, remain blocked by missing PyTorch and active/partly allocated GPU use. No output hashes, bitwise GPU agreement, timings, loop throughput, VRAM allocator peaks, host RSS during GPU execution, pinned bytes, or measured transfer bandwidth exist for this run. The separate host-DRAM prototype is also blocked because it requires CUDA to verify actual residency and transfer costs; a CPU substitute would not answer that question.

**Next decisive experiment:** when the RTX 5080 is idle, install or select a CUDA-compatible PyTorch build in an isolated environment, rerun preflight, then start with compiled NAND smoke and resident transformer FP32/BF16/FP16 correctness. Only after equality passes, run the prescribed resident and 1/8/32-roundtrip cases with five repetitions. Then implement the isolated active-session DRAM prototype and compare measured concurrent states against a clean resident baseline, including the random pointer-chasing control.

Raw numerical cases: `numerical_stability_cpu.json`. Executable: `../experiments/numerical_stability.py`.
