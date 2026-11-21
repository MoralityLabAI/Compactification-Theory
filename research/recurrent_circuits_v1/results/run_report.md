# RC-01 recurrent-circuit compactification experiment report

Pinned source: `8a79b9668d2fc4f2e2028b8914379aded33285f7`. Upstream `main` matched that pin at fetch. Local branch: `research/rc01-5080-experiments`.

## Preflight and reference gate

- `python -m unittest discover -v`: 23/23 passed.
- `python -O -m unittest discover -v`: 23/23 passed.
- `python verify.py --check results/reference_v1.json` and its optimized run: source-hashed receipt verified.
- Windows initially checked out CRLF files while the receipt hashes raw bytes. Restoring exact Git blob bytes made the receipt pass; pinned source content was not changed.
- RTX 5080 Laptop GPU (compute capability 12.0), driver 577.05 / CUDA API 12.9, Python 3.12.10. CUDA runs used isolated task-local PyTorch 2.11.0+cu128 (runtime 12.8), NumPy 2.5.4, and psutil 7.2.2. The official PyTorch release matrix lists the Windows Python 3.12 CUDA 12.8 wheel: [PyTorch releases](https://github.com/pytorch/pytorch/releases).
- This is a shared Windows WDDM device. The process owner was not visible. `nvidia-smi`, its command-line memory query, and PyTorch `mem_get_info` disagreed, and readings varied significantly. Each run records its PyTorch process allocator peak and cap. Host RAM also ran high; available system RAM was about 1.0–1.6 GiB during the multitenant run and process peak working set about 1.04 GB. Snapshots are in `preflight_hardware.json` and raw run files.
- CUDA jobs were serialized. NAND used a 1 GiB allocator cap; transformer runs used 2 GiB; multitenant used an 8% per-process cap (about 1.27 GiB). No other process was stopped.

## CUDA NAND

Fixed tensor NAND, `--confirm-gpu`, full shape batch 8 × 512 tokens × width 256, 128 loops, one warmup, five timed repeats. Independent Boolean replay passed for all dtypes.

| Precision | Median | Loops/s | Peak allocated / reserved | Oracle |
|---|---:|---:|---:|---|
| FP32 | 0.110 s | 1,162.6 | 28 / 40 MiB | pass |
| BF16 | 0.070 s | 1,816.9 | 14 / 20 MiB | pass |
| FP16 | 0.067 s | 1,897.7 | 14 / 20 MiB | pass |

This establishes CUDA correctness for the 128-step fixed circuit. The 2048-step reach result below is CPU scalar emulation.

## Untrained transformer: resident and explicit roundtrips

Configuration: batch 8, tokens 512, width 256, loops 128; FP32, BF16, FP16; resident and spill intervals 1, 8, 32; one warmup, five timed repeats. Every spill result matched its same-precision resident result bitwise. Raw JSON files contain output hashes, timing samples, allocator peaks, traffic, and source hashes. The spill implementation retained the full state allocation, so allocated/reserved peaks were identical to resident execution at each dtype.

| Precision | Resident median (range) | Spill k=1 | Spill k=8 | Spill k=32 |
|---|---:|---:|---:|---:|
| FP32 | 0.950 s (0.260–1.087) | 0.291 s | 0.232 s | 0.227 s |
| BF16 | 0.186 s (0.181–0.222) | 0.198 s | 0.277 s | 0.183 s |
| FP16 | 0.242 s (0.195–0.327) | 0.393 s | 0.272 s | 0.323 s |

FP32 resident timings varied widely while spill samples were narrower; BF16 and FP16 also varied. The GPU was concurrently used by an unowned WDDM process. These data establish correctness and traffic, but do not support a clean speed comparison across modes.

| Precision | State / pinned buffer | H2D and D2H per repeat, k=1 | k=8 | k=32 | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|
| FP32 | 4 MiB | 512 MiB each direction | 64 MiB | 16 MiB | 72 / 86 MiB |
| BF16 | 2 MiB | 256 MiB each direction | 32 MiB | 8 MiB | 52 / 56 MiB |
| FP16 | 2 MiB | 256 MiB each direction | 32 MiB | 8 MiB | 52 / 56 MiB |

At k=1, BF16 moved from 0.186 s resident to 0.198 s, and FP16 from 0.242 s to 0.393 s. Other pairs were noisy and sometimes faster than resident; this is consistent with shared-GPU timing variation, not evidence that transfers improve execution. The explicit spill arms saved **zero VRAM**.

## Host-DRAM multitenant prototype

A separate prototype kept 16 BF16 session states (2 MiB each; 32 MiB total) in pinned host memory. It moved active microbatches of 1, 4, or 8 sessions to the GPU, ran 32 transitions, and tested 1, 8, or 32 loops per residence. All nine host-backed outputs matched the resident reference bitwise for the same active batch shape.

| Active sessions | Resident peak allocated / reserved | Host-backed peak allocated / reserved | Host state pinned | k=1 slowdown (time) | k=8 | k=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58.6 / 76 MiB | 28.6 / 44 MiB | 32 MiB | 1.61× (0.813 s) | 0.74× (0.373 s) | 0.59× (0.298 s) |
| 4 | 106.6 / 164 MiB | 82.6 / 120 MiB | 32 MiB | 2.07× (0.447 s) | 1.12× (0.242 s) | 1.00× (0.215 s) |
| 8 | 170.6 / 280 MiB | 154.6 / 264 MiB | 32 MiB | 1.74× (0.361 s) | 1.00× (0.209 s) | 0.92× (0.191 s) |

Peak allocated GPU memory fell by 30, 24, and 16 MiB at active counts 1, 4, and 8. Inactive states stayed in the 32 MiB host pool. This is measured residency savings for this small pool; both modes fit, so it does not establish a higher maximum session count.

At k=1, H2D and D2H traffic were each 1 GiB for all 16 sessions; at k=8 each direction was 128 MiB; at k=32 each direction was 32 MiB. The pinned state pool was 32 MiB. k=1 slowed all three active counts. At k=8/32 the results approached or fell below resident, but this prototype has one timing per arm on a shared GPU; the active=1 inversions are not reliable speedups. A randomized paired repeat at idle is needed to locate the transfer amortization point.

The random pointer-chasing control used a 1,048,576-entry (4 MiB) host table for 128 dependent hops. Each hop moved one 4-byte address in each direction and required a CPU lookup before the next H2D transfer. It matched its CPU oracle and measured 11,967 dependent hops/s. This serialized control did not prefetch or overlap transfers.

A resident-pool control requested 653 BF16 session states (1,369,440,256 bytes) under the explicit allocator cap and reported CUDA OOM without reducing the batch. This is a per-process cap control, not physical VRAM exhaustion; host backing was not run at 653 sessions because system RAM was limited.

## Recurrent numerical stability

CPU scalar software emulation rounded every modeled primitive to FP32, BF16, or FP16 and checked every intermediate transition against an independent integer oracle. With restored NAND and coherent +1/8 perturbation, all 16 width-4 initial states and five width-16 seeds remained correct through 2048 transitions. Disabling gate restoration failed at transition 5; leaving idle cells unrefreshed failed at transition 4 for widths 16, 64, and 256. Refreshing idle cells preserved correctness through 2048.

These outcomes agree with Theorem 5's conditional bound for this circuit: separation 1, radius 1/8, restored symbolic map rho=0, and per-step perturbation eta=1/8, so 2r<Delta and eta<=(1-rho)r. The result does not establish these assumptions for learned weights or CUDA arithmetic.

## Answers and scope

1. **Fixed weights and preserved symbolic behavior:** the constructed NAND recurrence was exact through 2048 steps in CPU emulation, and the CUDA NAND oracle passed at 128 steps. The transformer was untrained and makes no symbolic-circuit claim.
2. **Stable restoration:** tested CPU precision emulations agreed with the theorem assumptions; removing gate restoration or idle refresh caused early failures. This does not prove a general learned-model horizon.
3. **Host backing and capacity:** the prototype demonstrated inactive host-resident states and lower GPU allocator peaks for 16 sessions. It did not show a same-budget resident OOM and successful host-backed run at the same count, so a maximum capacity gain is not established.
4. **Transfer overhead:** k=1 incurred measured slowdown in the multitenant prototype. Longer residence reduced bytes 8×/32× and usually approached resident time, but shared-device timing inversions remain. The supplied spill benchmark itself saved no VRAM.

**Next decisive experiment:** repeat resident and host-backed arms in randomized paired order when the GPU is idle, increase the fixed session pool until a resident allocation reaches a declared process budget, then run that same pool through host backing. Keep the 8×512×256 BF16 block, record completed sessions, RSS, pinned bytes, allocator peaks, and transfer timings, and rerun the dependent pointer-chasing control.

Artifacts: raw CUDA JSONs and `gpu_matrix_summary.json`; host prototype `../experiments/multitenant_dram.py` / `multitenant_dram_bf16.json`; CPU numerical experiment `../experiments/numerical_stability.py` / `numerical_stability_cpu.json`; failed-attempt notes in `gpu_attempts.json`.
