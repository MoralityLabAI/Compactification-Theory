#!/usr/bin/env python3
"""Deterministic builder for the meta_sft v0.1 corpus.

Reads the pinned sources named in configs/corpus_sources.json through
`git show <commit>:<path>` (never the working tree), checks every quoted
evidence string against the pinned file text, renders SFT messages, splits by
lineage and writes train/val/heldout JSONL plus manifest.json.

Same inputs produce byte-identical outputs: no clocks, no environment-dependent
ordering, a fixed seed for the single random split.

Usage:  python build_corpus.py [--out DIR] [--check]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # experiments/bonsai-gym
CONFIG = ROOT / "configs" / "corpus_sources.json"
OUT_DEFAULT = ROOT / "datasets" / "meta_sft_v0_1"

SCHEMA_VERSION = "meta_sft/0.1"
SEED = 20261102
VAL_FRACTION = 0.10
MAX_SYNTHETIC_FRACTION = 0.30
DEDUP_JACCARD = 0.90

FAMILIES = [
    "task_graph_navigation", "curriculum_selection", "learning_gain_prediction",
    "skill_composition", "experiment_design", "failure_diagnosis",
    "specialist_routing", "quantization_aware_adaptation", "transfer_across_families",
]
HELDOUT_FAMILIES = ["transfer_across_families", "specialist_routing"]
EVIDENCE_TYPES = ["observed", "hypothesis", "synthetic"]
PREDICTION_BASES = ["preregistered", "mechanism_rationale", "proposed", "none"]

SYSTEM_PROMPT = (
    "You are the experiment controller of a local training gym that fine-tunes LoRA adapters "
    "on a low-bit base model. Given the task context and the available skills, propose the next "
    "action, state the predicted improvement, and say how it will be verified. Report only "
    "results that the context supports; never present a prediction as a measurement."
)

FAMILY_QUESTION = {
    "task_graph_navigation": "Which task or edge of the task graph should be worked on next, and what does the graph structure imply for how results may be read?",
    "curriculum_selection": "How should training exposure be allocated across tasks or stages, and what should be guarded against?",
    "learning_gain_prediction": "What gain do you predict for this action, and on what basis?",
    "skill_composition": "How should these skills be combined, and what does each component have to guarantee?",
    "experiment_design": "How should the experiment be designed or amended so that its result can be trusted?",
    "failure_diagnosis": "What is the most likely cause of this failure, and which test discriminates between the explanations?",
    "specialist_routing": "Which specialist or control policy should handle this input, and when should the router abstain?",
    "quantization_aware_adaptation": "What does the low-bit or quantized setting change about how the adapter should be trained, evaluated or served?",
    "transfer_across_families": "Will this result carry over to a different task family, adapter or base, and how should that be tested?",
}


def find_git():
    for name in ("GIT", "GIT_EXE"):
        if os.environ.get(name):
            return os.environ[name]
    found = shutil.which("git")
    if found:
        return found
    pat = os.path.join(os.environ.get("LOCALAPPDATA", ""), "GitHubDesktop", "app-*",
                       "resources", "app", "git", "cmd", "git.exe")
    hits = sorted(glob.glob(pat))
    if hits:
        return hits[-1]
    raise SystemExit("git not found: set GIT or put git on PATH")


GIT = find_git()


def load_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {s["key"]: s for s in cfg["sources"]}


SOURCES = load_config()
_TEXT_CACHE = {}
_NORM_CACHE = {}


def git_show(src, path):
    k = (src, path)
    if k in _TEXT_CACHE:
        return _TEXT_CACHE[k]
    cfg = SOURCES[src]
    r = subprocess.run([GIT, "-C", cfg["local_path"], "show", "%s:%s" % (cfg["commit"], path)],
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit("cannot read %s:%s in %s (clone %s into %s): %s" % (
            cfg["commit"][:12], path, cfg["local_path"], cfg.get("remote_url", "?"),
            cfg["local_path"], r.stderr.decode("utf-8", "replace").strip()))
    text = r.stdout.decode("utf-8")
    _TEXT_CACHE[k] = text
    return text


def _norm_index(src, path):
    k = (src, path)
    if k not in _NORM_CACHE:
        text = git_show(src, path)
        kept, idx = [], []
        for i, ch in enumerate(text):
            if ch.isspace() or ch in "*`":
                continue
            kept.append(ch)
            idx.append(i)
        _NORM_CACHE[k] = ("".join(kept), idx, text)
    return _NORM_CACHE[k]


def norm(s):
    return "".join(ch for ch in s if not (ch.isspace() or ch in "*`"))


def locate(src, path, quote):
    """Return (first_line, last_line) of `quote` in the pinned file, or raise."""
    ntext, idx, text = _norm_index(src, path)
    nq = norm(quote)
    if not nq:
        raise ValueError("empty quote")
    pos = ntext.find(nq)
    if pos < 0:
        raise ValueError("quote not found in %s:%s: %r" % (src, path, quote))
    a = idx[pos]
    b = idx[pos + len(nq) - 1]
    return text.count("\n", 0, a) + 1, text.count("\n", 0, b) + 1


SPECS = []
ERRORS = []


def R(lin, fam, ev, src, path, sec, q, ctx, sk, act, pred, *rest):
    """Register one evidence-backed record spec.

    q: verbatim evidence strings (whitespace, * and ` are ignored when matching);
       an item may be (other_path, text) to cite a second file of the same source.
    pred/out: None when the source states no prediction / no result.
    """
    # observed: (out, ver, pb); hypothesis: (ver, pb) with out = None
    if ev == "hypothesis":
        out = None
        ver, pb = rest
    else:
        out, ver, pb = rest
    assert fam in FAMILIES, fam
    assert ev in ("observed", "hypothesis"), ev
    assert pb in PREDICTION_BASES, pb
    SPECS.append(dict(lin=lin, fam=fam, ev=ev, src=src, path=path, sec=sec, q=list(q), ctx=ctx,
                      sk=list(sk), act=act, pred=pred, out=out, ver=ver, pb=pb))


SYNTH = []


def S(lin, fam, name, basis, basis_q, ctx, sk, act, pred, ver):
    """Register one templated synthetic spec. basis=(src, path) is the real file whose
    stated procedure the template follows; basis_q are verbatim strings from it."""
    SYNTH.append(dict(lin=lin, fam=fam, name=name, basis=basis, basis_q=list(basis_q), ctx=ctx,
                      sk=list(sk), act=act, pred=pred, ver=ver))


# ---------------------------------------------------------------------------
# Compactification-Theory (adapter-bits) main: adapter capacity follow-ups
# ---------------------------------------------------------------------------
SK_AB = ["gate LoRA adapter at chosen layers and ranks", "full-batch plain gradient descent or Adam",
         "Fano estimate of stored label bits I = M(1 - H2(err))", "sharpness (top Hessian eigenvalue) tracking",
         "post-training rounding to b bits per weight"]
AB = "ab_main"
FU = "docs/followups.md"
CTX_A = ("Capacity study on a 27-layer Qwen-style model: random label bits (M items) are stored in a rank-1 last-layer "
         "gate LoRA (W = 4,096 trainable parameters) and stored information is estimated with the Fano formula. "
         "Pre-registered follow-ups were pushed on 2026-10-12 before any of them ran.")

R("adapter-bits/followups-A1", "experiment_design", "observed", AB, FU, "Calibration that shaped the design (gd-calib-2)",
  ["Sharpness rose to 2/η = 20 and stayed there", "the adapter stored 11 bits", "storing 5 and 3 bits",
   "comparisons across step sizes are made at equal gradient-flow time T = η·steps = 400"],
  CTX_A + " A calibration run (gd-calib-2) used full-batch gradient descent for 4,000 steps at eta = 0.1, 0.01 and 0.001 with M = 256 random labels.",
  SK_AB,
  "Compare step sizes at equal gradient-flow time T = eta x steps = 400 (4,000, 13,333, 40,000 and 133,333 steps for eta 0.1, 0.03, 0.01, 0.003) instead of at equal step count.",
  None,
  "At eta = 0.1 sharpness rose to 2/eta = 20 and stayed there, and the adapter stored 11 bits. At eta = 0.01 and 0.001 the runs were far from the stability limit (eta*lambda = 0.20 and 0.007 at the end), simply slow, storing 5 and 3 bits. So the source compares step sizes at equal gradient-flow time.",
  "Track eta*lambda at every logged point; a fair comparison needs each run either at the stability limit or to be labelled as not yet there.",
  "none")

R("adapter-bits/followups-A1", "learning_gain_prediction", "observed", AB, FU, "A1, P-A1a",
  ["P-A1a: passes.", "η·λ was 1.87 to 2.35 at every tracked point after first reaching the limit",
   "In any run whose tracked sharpness reaches 1.5/η, every later tracked value stays within [1.5, 2.6]/η"],
  CTX_A + " A1 sets layer 27, gate LoRA rank 1, M = 256, full batch, T = 400, with eta in {0.1, 0.03, 0.01, 0.003}.",
  SK_AB,
  "Register P-A1a (edge of stability, Cohen et al. 2021): in any run whose tracked sharpness reaches 1.5/eta, every later tracked value stays within [1.5, 2.6]/eta.",
  "Registered before the run: sharpness settles at the stability limit 2/eta.",
  "P-A1a passes. Every run's sharpness climbed to the stability limit and stayed there; eta*lambda was 1.87 to 2.35 at every tracked point after first reaching the limit, and the final sharpness was 2/eta within 5% at every eta. Before reaching the limit, sharpness rose steadily (progressive sharpening).",
  "Log sharpness at fixed intervals; count tracked points inside [1.5, 2.6]/eta after the run first reaches 1.5/eta (7 of 7, 12 of 12, 15 of 15 and 18 of 18 in the four runs).",
  "preregistered")

R("adapter-bits/followups-A1", "learning_gain_prediction", "observed", AB, FU, "A1, P-A1b",
  ["P-A1b: fails as registered.", "I(0.03) = 30.1 exceeds I(0.01) = 18.3 by 11.8 bits",
   "I(0.003) ≥ I(0.01) ≥ I(0.03) ≥ I(0.1), each within 10 bits of tolerance"],
  CTX_A + " A1 compares stored bits at equal gradient-flow time T = 400 for four step sizes.",
  SK_AB,
  "Register P-A1b (conjecture): stored bits at T = 400 do not increase with eta, I(0.003) >= I(0.01) >= I(0.03) >= I(0.1), each within 10 bits of tolerance.",
  "Stored bits are non-increasing in the step size, within 10 bits.",
  "P-A1b fails as registered: I(0.03) = 30.1 exceeds I(0.01) = 18.3 by 11.8 bits, beyond the 10-bit tolerance; the other two pairs hold. The eta = 0.03 run's error fell from 0.398 at step 13,000 to 0.301 at its final step 13,333 while eta*lambda jumped from 1.87 to 2.07; the final iterate of a run oscillating at the edge of stability is a noisy sample, but the registered rule used it and the failure stands.",
  "Score the registered rule on the final iterate as written; if an iterate-averaging rule seems better, record it as a post-hoc amendment rather than rescoring.",
  "preregistered")

R("adapter-bits/followups-A1", "learning_gain_prediction", "observed", AB, FU, "A1, P-A1c",
  ["P-A1c: passes.", "stored 45 bits against Adam's 256", "Adam's memorizing solution has λ ≈ 1,400, above 2/η = 200"],
  CTX_A + " Adam (lr 0.03) memorized all 256 labels on the same setup at sharpness about 1,400.",
  SK_AB,
  "Register P-A1c (conjecture): no gradient-descent run with eta >= 0.01 memorizes all 256 labels, because Adam's memorizing solution has lambda about 1,400, above 2/eta = 200.",
  "No run with eta >= 0.01 memorizes all 256 labels.",
  "P-A1c passes. No run memorized: stored bits were 10.9, 30.1, 18.3 and 45.2 for eta 0.1, 0.03, 0.01 and 0.003; even eta = 0.003, whose limit 2/eta of about 667 is closest to Adam's lambda of about 1,400, stored 45 bits against Adam's 256.",
  "Compare final stored bits against the 256-label ceiling and check the final sharpness against 2/eta.",
  "preregistered")

R("adapter-bits/followups-A2", "learning_gain_prediction", "hypothesis", AB, FU, "A2",
  ["Layer 24 alone, gate LoRA rank 4 (`24:4`), M = 256, full batch, T = 400, η ∈ {0.1, 0.03}.",
   "layer 24 stores at least 4 times the bits that layer 27 stores in A1 at the same η"],
  CTX_A + " Layer-24 solutions are flat (lambda about 5 to 50) while last-layer solutions that store labels are sharp (lambda about 1e3 under Adam).",
  SK_AB,
  "Run A2: layer 24 alone, gate LoRA rank 4, M = 256, full batch, T = 400, eta in {0.1, 0.03}, and compare with the layer-27 A1 runs at the same eta.",
  "P-A2 (conjecture): at each eta, layer 24 stores at least 4 times the bits that layer 27 stores in A1, because flat high-capacity solutions exist there.",
  "Compare I(layer 24) against 4 x I(layer 27, same eta) at equal gradient-flow time; report the ratio per eta.",
  "preregistered")

R("adapter-bits/followups-B", "learning_gain_prediction", "hypothesis", AB, FU, "B2, P-B4",
  ["Depth, pool 2,048, M = 2,048, 4,000-step cap, specs `27:4`, `24:4` and `24:1,25:1,26:1,27:1`.",
   "At M = 2,048, `24:4` and the four-layer adapter each store at least 5 times the bits of `27:4`."],
  CTX_A + " Seeds 1 and 2 get new random keys, labels and initializations with Adam at the original settings.",
  SK_AB,
  "Replicate the depth comparison on two new seeds: pool 2,048, M = 2,048, 4,000-step cap, adapters 27:4, 24:4 and 24:1,25:1,26:1,27:1.",
  "P-B4 (conjecture, replicating deep-2/3): at M = 2,048, 24:4 and the four-layer adapter each store at least 5 times the bits of 27:4, and the four-layer adapter stores at least as much as 24:4 minus 50 bits.",
  "Count stored bits per seed and adapter; the prediction is judged per seed against the 5x and minus-50-bit margins.",
  "preregistered")

R("adapter-bits/followups-B", "learning_gain_prediction", "hypothesis", AB, FU, "B1, P-B2 and P-B3",
  ["Probe k = 64 stores at least 1.5 bits per parameter at M = 128.",
   "Gate LoRA rank 8 (W = 32,768) stores less than 3 times the bits of rank 1 (W = 4,096) on each seed."],
  CTX_A + " Single-layer probes and gate LoRAs are rerun on seeds 1 and 2 (pool 4,096, 3,000 Adam steps).",
  SK_AB,
  "Run probe k = 64 at M = 128, probe k = 512 at M = 1,024, gate LoRA rank 1 at M = 256 and gate LoRA rank 8 at M = 512 on seeds 1 and 2.",
  "P-B2 (Cover's theorem): probe k = 64 stores at least 1.5 bits per parameter at M = 128 and probe k = 512 at least 0.8 bits per parameter at M = 1,024. P-B3 (conjecture): rank 8 stores less than 3 times the bits of rank 1 on each seed.",
  "Compute I/W for each probe run and the rank-8 to rank-1 stored-bits ratio per seed.",
  "preregistered")

R("adapter-bits/followups-C", "learning_gain_prediction", "hypothesis", AB, FU, "C, P-C1 and P-C2",
  ["last-token gate LoRA rank 1 in each of layers 24 to 27 (W = 16,384), M = 16,384, Adam lr 0.03, cap of 12,000 steps",
   "If that rank reaches the 2,048-row cap, the test is inconclusive and is reported as such."],
  CTX_A + " The four-layer rank-1 adapter (W = 16,384) is pushed to M = 16,384 items to find where it saturates; gradient rows are sampled on 2,048 items so the rank estimate can reach 2,048.",
  SK_AB,
  "Run the saturation test: last-token gate LoRA rank 1 in each of layers 24 to 27, M = 16,384, Adam lr 0.03, 12,000-step cap, and read stored bits against the gradient-matrix rank.",
  "P-C1 (spectral-capacity reading): stored bits are at most 1.3 times twice the trained gradient matrix's rank at relative threshold 1e-3, and the test is declared inconclusive if the rank reaches the 2,048-row cap. P-C2: the knee is at most 9 bits per weight and the top Hessian eigenvalue is at most 100.",
  "Compute the rank at threshold 1e-3 on the 2,048-row sample, the rounding knee and the top Hessian eigenvalue after training.",
  "preregistered")

CTX_D = ("Gate LoRA adapters with the same trainable budget (W = 16,384) are compared at different depths and ranks on a 27-layer Qwen-style model; "
         "each stores M random label bits and is scored by the Fano estimate.")
R("adapter-bits/depth-sweeps", "experiment_design", "observed", AB, "results/deep-2/report.md", "deep-2 runs and rank table",
  ["|deep-gate|27:4|0.03|16384|2048|0.125|0.3667|0.0065|", "|deep-gate|24:1,25:1,26:1,27:1|0.03|16384|2048|0.125|0.0|0.125|",
   "|deep-gate|27:4|0.03|2048|106|512|", "|deep-gate|26:2,27:2|0.03|16384|2048|0.125|0.15332|0.0477|"],
  CTX_D + " Items M = 2,048, Adam lr 0.03.", SK_AB,
  "Spread the budget as rank 1 in each of layers 24, 25, 26 and 27 rather than concentrating rank 4 in layer 27 or rank 2 in layers 26 and 27.",
  "Spreading across depth should raise capacity at fixed W, because the gradient matrix gains rank (Cover's theorem caps a linearized adapter near twice that rank).",
  "At M = 2,048 the adapter 27:4 stored 106 bits (error 0.3667, top Hessian eigenvalue 139.7), 26:2,27:2 reached error 0.15332, and 24:1,25:1,26:1,27:1 stored all 2,048 (error 0.0, eigenvalue 10.05).",
  "Estimate stored bits with the Fano formula at the end of training and check the counting bound; report the gradient-matrix rank next to stored bits.",
  "mechanism_rationale")

R("adapter-bits/depth-sweeps", "experiment_design", "observed", AB, "results/deep-3/report.md", "deep-3 runs and rank table",
  ["|deep-gate|24:4|0.03|8192|848|512|770|1022|7.9|", "|deep-gate|24:1,25:1,26:1,27:1|0.03|8192|3284|512|846|1022|12.85|"],
  CTX_D + " Items are pushed from M = 2,048 to M = 8,192 to find where layer-24 rank 4 and the four-layer rank-1 adapter diverge.", SK_AB,
  "Raise M to 8,192 and compare a single rank-4 adapter at layer 24 with rank 1 at each of layers 24 to 27.",
  None,
  "At M = 2,048 both stored all 2,048 bits; at M = 8,192 the single-layer 24:4 stored 848 bits and the four-layer adapter 3,284 bits (best stored bits at W = 16,384).",
  "Report stored bits at two M values so that a saturated and an unsaturated regime are both visible.",
  "none")

R("adapter-bits/depth-sweeps", "quantization_aware_adaptation", "observed", AB, "results/deep-3/report.md", "Information per parameter after rounding",
  ["|deep-gate|24:1,25:1,26:1,27:1|0.03|16384|2048|0.125|0.0|0.125|7(sample7)|7|16|10.56|",
   "0.1199|0.1225|0.125|0.125|0.125", "0.0909|0.123|0.125|0.125|0.125"],
  CTX_D + " After training, every tensor is rounded to b bits (b = 1 to 16) and stored information I_b/W is re-measured; the counting bound caps it at b bits per parameter plus 32 bits per tensor for the scale.",
  ["post-training rounding to b bits", "typical-noise knee predictor", "worst-case l-infinity knee predictor", "Fano estimate of stored bits"],
  "Choose the storage precision by the rounding knee, the fewest bits per weight that keep error within one point of full precision, rather than by the training dtype.",
  "The knee is predicted from the gradient sample by a typical-noise model (7 bits) and a worst-case l-infinity model (16 bits).",
  "For the four-layer adapter at M = 2,048 the measured knee was 7 bits (typical-noise prediction 7, worst-case prediction 16): I_b/W was 0.1199 at b = 7, 0.1225 at b = 8 and 0.125 from b = 10, the full-precision value. The 24:4 adapter behaved alike (0.0909 at b = 7, 0.123 at b = 8, 0.125 at b = 10).",
  "Round each tensor with its own scale, re-run the label readout and compare against the prediction column; check that the counting bound held.",
  "mechanism_rationale")

R("adapter-bits/pilot-2", "experiment_design", "observed", AB, "results/pilot-2/report.md", "Capacity scaling",
  ["- lora:gate (lr 0.003): best stored bits by W = 4096: 25", "- lora:gate (lr 0.01): best stored bits by W = 4096: 44",
   "- lora:gate (lr 0.03): best stored bits by W = 4096: 62", "|lora:gate|1|0.03|4096|2048|0.5|0.39746|0.0153|"],
  "Rank-1 last-layer gate LoRA (W = 4,096) on M = 2,048 random labels; the pilot varies only the Adam learning rate.",
  SK_AB,
  "Sweep the Adam learning rate over 0.003, 0.01 and 0.03 before fixing it for the main grid.",
  None,
  "Best stored bits at W = 4,096 were 25 (lr 0.003), 44 (lr 0.01) and 62 (lr 0.03); at lr 0.03 the error was 0.39746 and I/W 0.0153, so the highest rate tried stored the most.",
  "Hold items, seed and adapter fixed and compare stored bits at the end of the same step budget.",
  "none")


# ---------------------------------------------------------------------------
# SIML-01 (task graph draft) and SIML-01N (measured neural curriculum pilot)
# ---------------------------------------------------------------------------
SI = "ab_siml"
SD = "research/self_introspective_meta_learning_v1/"
NR, NP, NG, RM = SD + "NEURAL_RESULTS.md", SD + "NEURAL_PROTOCOL.md", SD + "NEURAL_PROGRESS.md", SD + "README.md"
SK_CUR = ["fixed 5:9:6 curriculum over T0/T1/T4", "random task-family allocation", "greedy behavioral learning-progress scheduler",
          "greedy introspective scheduler (closure residual)", "guarded scheduler (35% diagnostic preference, 65% frozen family distribution)",
          "guarded scheduler with shuffled diagnostics"]
CTX_N = ("SIML-01N task graph with three trained nodes: T0 (eight-state hidden-bit world), T1 (compositions of T0 primitives) and T4 "
         "(modular addition with shear, a 31-way output). A width-48 tied GRU student is trained for 320 AdamW updates of 32 examples per arm on CPU; "
         "five new seeds 20-24; primary score is the family-weighted accuracy on T0/T1/T4 with weights 0.25/0.45/0.30; "
         "T2 three-action compositions are an unseen secondary horizon.")

R("siml01/neural-curriculum-5seed", "curriculum_selection", "observed", SI, NR, "Five-seed table",
  ["| Random | **0.5446** | 0.6500 | 0.0069 | 0.9167 |", "| Behavioral greedy | 0.4425 | 0.5000 | 0.0583 | 0.5000 |",
   "| Introspective greedy | 0.3519 | 0.4875 | 0.0875 | 0.4917 |", "| Guarded introspective | 0.5325 | 0.6250 | 0.0042 | 0.9417 |",
   "| Fixed | 0.5048 | 0.5625 | 0.0056 | 0.8750 |", "Random selection remains strongest on the primary weighted score.",
   (NP, "A naive greedy policy is included as a failure control.")],
  CTX_N, SK_CUR,
  "Allocate the 320 updates with the guarded scheduler (positive coverage of every task family) rather than an unconstrained greedy scheduler, and keep random allocation as the baseline to beat.",
  "The protocol registers the naive greedy policy as a failure control: a difficult and initially unlearnable task must not capture every update, so guarded coverage should avoid the starvation.",
  "Weighted T0/T1/T4 accuracy over five seeds: random 0.5446, guarded introspective 0.5325, fixed 0.5048, behavioral greedy 0.4425, introspective greedy 0.3519. Guarding recovered most of the greedy loss but did not beat random selection, which stayed strongest on the primary score.",
  "Report arm-wise seed values and signed pairwise differences, not only the five-seed mean, and keep the primary endpoint (weighted T0/T1/T4) fixed before looking at the T2 secondary.",
  "preregistered")

R("siml01/neural-curriculum-5seed", "failure_diagnosis", "observed", SI, NR, "Observed behavior",
  ["Both greedy strategies concentrate training on T4's difficult 31-way arithmetic, reducing aggregate performance.",
   "| Introspective greedy | 0.3519 | 0.4875 | 0.0875 | 0.4917 |"],
  "Two greedy curricula (behavioral learning-progress and introspective closure residual) score 0.4425 and 0.3519 weighted accuracy against 0.5048 for the fixed curriculum, with T2 unseen-composition accuracy 0.5000 and 0.4917 against 0.8750.",
  SK_CUR + ["per-family update counts from the allocation trace"],
  "Read the allocation traces: count the updates each arm gave to T0, T1 and T4 and check whether the loss is starvation of established competencies.",
  None,
  "Both greedy strategies concentrated training on T4's difficult 31-way arithmetic, which reduced aggregate performance. T4 held-out accuracy rose slightly (0.0583 and 0.0875 vs 0.0056 for the fixed arm) while T1 fell to 0.5000 and 0.4875, so established competencies lost ground to a task that stayed near the floor.",
  "Compare per-family allocation counts and per-family held-out accuracy across arms; a guarded arm with enforced positive coverage is the control.",
  "none")

R("siml01/neural-curriculum-5seed", "experiment_design", "observed", SI, NR, "Observed behavior and shuffled falsifier",
  ["The shuffled diagnostic arm slightly exceeds the measured introspective arm", "there is no observed incremental advantage from the closure residual in this study",
   "| Guarded shuffled diagnostic | 0.5381 | 0.6375 | 0.0042 | **0.9500** |",
   (NP, "The critical diagnostic ablation is")],
  CTX_N + " The question is whether the measured hidden-state closure residual adds anything beyond the same scheduler reading shuffled residual values.",
  SK_CUR,
  "Run the guarded scheduler a second time with a fixed permutation of the closure-residual values across task families (same call pattern, same diagnostic cost) and compare it with the measured-residual arm.",
  "Registered as the critical diagnostic ablation: if introspection helps, the measured arm should beat the shuffled arm.",
  "The shuffled arm scored 0.5381 weighted accuracy against 0.5325 for the measured guarded arm, and 0.9500 against 0.9417 on T2. There was no observed incremental advantage from the closure residual in this study.",
  "Keep identical architecture, optimizer, initial checkpoint per seed and diagnostic rounds; only the mapping from residual to task family differs.",
  "preregistered")

R("siml01/neural-curriculum-5seed", "experiment_design", "observed", SI, NR, "Resource accounting",
  ["Each arm receives 320 updates, 10,240 sampled training examples and 20,480 recurrent step-example operations per seed",
   "A FLOP profiler and fully matched auxiliary-target acquisition cost will be necessary for strict equal-total-compute claims."],
  CTX_N, SK_CUR,
  "Match arms on updates, sampled examples and recurrent step-example operations, report elapsed time per run, and state which costs are not matched.",
  None,
  "Each arm received 320 updates, 10,240 sampled training examples and 20,480 recurrent step-example operations per seed. Task-specific probe operations and the number of privileged labels differed by curriculum, so strict equal-total-compute claims need a FLOP profiler and matched auxiliary-target acquisition cost. No proposer LLM inference or paid compute was used.",
  "Tabulate matched and unmatched costs per arm in the receipt; do not describe the comparison as compute-matched beyond what is tabulated.",
  "none")

R("siml01/neural-curriculum-5seed", "learning_gain_prediction", "observed", SI, NR, "Prior diagnostic follow-up",
  ["the difficult shear task reached 0.382 held-out accuracy (55/144 correct)", "In a separate exploratory 1,600-update run focusing mostly on T4 (seed 14)"],
  "T4 (modular addition with shear) stays near the floor in the 320-update arms (held-out accuracy at most 0.0875 in any arm). A separate exploratory run spends 1,600 updates mostly on T4 with seed 14.",
  SK_CUR,
  "Lengthen the T4 budget to 1,600 updates before concluding that the shear arithmetic is unlearnable at this width.",
  None,
  "In the separate exploratory 1,600-update run the shear task reached 0.382 held-out accuracy (55/144 correct). The short budget therefore does not support a claim that the arithmetic is unlearnable; the run follows the pilot and sits outside the frozen five-seed comparison.",
  "Report the exact count of held-out pairs (144 test pairs disjoint from 678 training and 139 development pairs) and label the run exploratory.",
  "none")

R("siml01/neural-curriculum-5seed", "task_graph_navigation", "observed", SI, NR, "Task splits",
  ["T0 is an eight-state sanity check; its worlds overlap train/dev/test.", "T4's 144 test state-pairs are disjoint from 678 training and 139 development pairs",
   "The 24 T2 three-action cases are unseen sequence lengths, using known primitives."],
  "Task graph T0 -> T1 -> T2 with T4 as an independent capacity diagnostic. T0 reuses its eight states in train, development and test; T1 holds out two action compositions but shares the eight microstates; T4 hashes (a,b) pairs into disjoint buckets; T2 length-three words were added after pilot inspection.",
  ["per-node held-out axis bookkeeping", "family-weighted primary endpoint", "secondary unseen-composition endpoint"],
  "Read T0 only as a sanity check, T1 as whole-composition generalization over shared microstates, T4 as a held-out-pair test, and T2 as a modest unseen-horizon check, and report them separately.",
  None,
  "The source records that T0 overlaps train/dev/test, T4's 144 test state-pairs are disjoint from 678 training and 139 development pairs, and the 24 T2 cases are unseen lengths using known primitives; their non-independence and small population limit how broadly the outcome generalizes.",
  "Fix splits by a label-independent hash and record, per node, what is held out and what is shared.",
  "none")

R("siml01/neural-curriculum-5seed", "experiment_design", "observed", SI, NG, "Evidence checks",
  ["49 unit tests pass", "30 trained checkpoint fingerprints were replayed on the original CPU runtime", "No 27B model was loaded, and no paid compute was used."],
  "A CPU-only PyTorch 2.10 reference for the neural curriculum comparison exists with a task graph, six selectors, a strict opt-in local 27B proposal-only interface and an offline source-hash and retrain verifier.",
  ["standard-library receipt verifier", "optional --retrain replay of checkpoint fingerprints", "unit tests in normal and optimized Python"],
  "Verify the result receipt offline: source hashes, all 30 run identities, task allocation, aggregate arithmetic and recurrent-work counts, then replay the 30 trained checkpoint fingerprints.",
  None,
  "49 unit tests passed in normal and optimized Python, and the 30 trained checkpoint fingerprints were replayed on the original CPU runtime. No 27B model was loaded and no paid compute was used.",
  "Run the verifier without network access; treat a fingerprint mismatch as a failed reproduction.",
  "none")

# --- SIML-01 task-graph design (proposals, no results) ---
R("siml01/task-graph-design", "task_graph_navigation", "hypothesis", SI, RM, "3. What the graph contains, Initial graph",
  ["T0 -> T1 -> T2 is the first local path.", "T3 and T4 provide independent capacity diagnostics.",
   "The T5/T2 -> T6 composition edge is a hypothesis.", "T7 waits for an adapter that preserves its declared observation and timing contract."],
  "Task graph of eight nodes: T0 hidden-bit world, T1 compositions of T0 primitives, T2 longer recurrent execution, T3 affine binary circuits, T4 modular addition with shear, T5 symbolic Storyworld rules, T6 routed modules solving conjunctions of learned skills, T7 partially observed timed planning via an IRC/Pluto adapter. The folder supplies no generators or model outcomes for the later nodes.",
  ["task-graph node contracts with generator and scorer hashes", "prerequisite, transfer, composition and interference edges", "frozen per-epoch target distribution"],
  "Work the path T0 -> T1 -> T2 first, use T3 and T4 as independent capacity diagnostics, then T5; treat T5/T2 -> T6 as an untested composition edge and keep T7 blocked until an adapter preserves its observation and timing contract.",
  "Not a numeric prediction: the ordering is a proposed navigation policy whose edges are testable claims.",
  "Each edge is a claim to be tested with a contemporaneous control; an unmeasured transfer edge stays labelled a hypothesis.",
  "proposed")

R("siml01/task-graph-design", "task_graph_navigation", "hypothesis", SI, RM, "3. What the graph contains, edge types",
  ["Restrict the prerequisite subgraph to a DAG; the transfer graph can contain cycles.", "An unmeasured transfer edge remains a hypothesis.",
   "Increasing horizon and adding a new intervention are separate edits."],
  "A task graph has prerequisite edges (testable claims about useful preparation), signed transfer edges from contemporaneous comparisons, composition hyperedges for skills that need several parents, interference edges and interface-expansion edges. Difficulty has several coordinates.",
  ["edge-type bookkeeping", "difficulty coordinates (horizon, intervention, interface)"],
  "Restrict the prerequisite subgraph to a DAG, allow cycles only in the transfer graph, and make every edge say which difficulty coordinate changed and which semantics stay fixed.",
  "Not a numeric prediction.",
  "A known quotient can certify an information requirement in an exact fixture; an unmeasured transfer edge is a hypothesis, not a fact about the graph.",
  "proposed")

R("siml01/task-graph-design", "task_graph_navigation", "hypothesis", SI, RM, "9. Learning the experiment policy",
  ["A learned graph edge also needs a contemporaneous control.",
   "Clone the same incumbent and compare training on the proposed prerequisite with matched alternative training, then score the target."],
  "A controller wants to record that training on task A is a prerequisite for task B in the learned graph. The evolving model's before-and-after scores mix transfer with its other updates.",
  ["clone of the incumbent checkpoint", "matched alternative training", "paired scoring on the target node"],
  "Clone the same incumbent, train one clone on the proposed prerequisite and another on matched alternative training, then score the target node on both.",
  "Not a numeric prediction: the edge is admitted only if the prerequisite clone beats the matched-alternative clone on the target.",
  "Compare the two clones on fresh target instances with the same update count, storage and optimizer budget.",
  "proposed")

DIAG_MAP = [
    ("Same code, different relevant futures", "Add a state coordinate or change the encoder", "Equal total code-bit budget alternative"),
    ("Initial state decodes; successor composition fails", "Add state-consistency loss or repair the transition", "Task-only training with matched updates"),
    ("Low adapter feature diversity at the chosen site", "Move or spread a fixed parameter budget", "Same data and training work"),
    ("Error grows with recurrent horizon", "Change precision, restoration or memory refresh", "Matched logical work and complete traffic accounting"),
    ("A dormant variable appears after an intervention", "Extend the task's represented state", "Present-only and intervention-aware versions"),
    ("An update damages an earlier family", "Add replay, isolate a module or revise routing", "Same total storage and optimizer budget"),
    ("Performance sits at the interface oracle ceiling", "Propose a separately versioned interface task", "Same-interface comparison retained"),
    ("A diagnostic fires before any learning", "Run random-init and shuffled-feature falsifiers", "Identical diagnostic fitting budget"),
]
for _e, _c, _ctl in DIAG_MAP:
    R("siml01/diagnosis-map", "failure_diagnosis", "hypothesis", SI, RM, "4. Diagnosis-to-experiment map",
      ["| %s | %s | %s |" % (_e, _c, _ctl)],
      "A learner in the SIML-01 task graph has been instrumented and the measured self-model shows: %s." % _e.lower(),
      ["held-out development loss by family", "one-step closure residual", "multi-step composition error", "near-fiber witnesses",
       "effective gradient rank at declared tolerances", "forgetting and measured cost"],
      "%s." % _c,
      "Not a numeric prediction: the source lists the experiment as the candidate response to this evidence.",
      "Required control: %s." % _ctl.lower(),
      "proposed")

R("siml01/experiment-policy", "curriculum_selection", "hypothesis", SI, RM, "9. Learning the experiment policy",
  ["An example budget split is 70% promising frontier updates, 20% retention/replay and 10% randomized diagnostic experiments.",
   "Start by comparing C and E after an engineering smoke run. Add D before attributing any gain to internal diagnostics."],
  "A controller schedules updates over a task graph. Arms: A fixed human curriculum, B random eligible task, C development learning-progress scheduler, D the proposer with shuffled or masked internal diagnostics, E measured self-introspective proposer, F E plus a trained experiment-policy adapter.",
  ["fixed human curriculum", "random eligible task and update", "development learning-progress scheduler", "measured introspective proposer", "shuffled-diagnostic control"],
  "Split the budget 70% promising frontier updates, 20% retention and replay, 10% randomized diagnostic experiments; compare arm C with arm E first and add arm D before crediting any gain to internal diagnostics.",
  "Not a numeric prediction; the split is a tunable draft default to be frozen for each comparison.",
  "Charge proposer tokens and diagnostic extraction to every arm even when they come from a subscription, and use paired independent training seeds for method comparisons.",
  "proposed")

R("siml01/experiment-policy", "learning_gain_prediction", "hypothesis", SI, RM, "4. The learner and its measured self-model",
  ["Measure forecast quality on subsequent development outcomes with a proper score, such as squared error for a bounded predicted gain.",
   "Log every attempted update, including rejected candidates.",
   "an archive of chosen actions alone does not establish their counterfactual value."],
  "Before each update the proposer writes a structured record: incumbent hash, target node, hypothesis, diagnostic receipts, competing explanation, update operator, training budget, predicted primary gain, predicted transfer, predicted retention, predicted cost, uncertainty and a discriminating test.",
  ["structured proposal record", "development outcomes after the update", "randomized shadow experiments"],
  "Score each forecast of primary gain against the subsequent development outcome with a proper score such as squared error, and log every attempted update including rejected ones.",
  "Not a numeric prediction: forecast quality is to be measured, and predictions about untried updates need randomized shadow experiments or another identified design.",
  "Fit the experiment-policy predictor on development outcomes and test it on fresh graph instances at matched total compute against the same proposer with a fixed policy.",
  "proposed")


# ---------------------------------------------------------------------------
# BGI-01B..H: instrument audits, verified repair, factored authorization, progress-selected repair
# ---------------------------------------------------------------------------
BG = "ab_bgi"
BD = "research/bgi_attractors_v1/results/"
BL = "bgi01/attractor-stress-chain"
SK_BGI = ["pinned small instruction-tuned model, CPU greedy generation", "frozen synthetic fixture with counterbalanced option order",
          "typed single-field JSON policy patch", "exact finite policy verifier and no-regression check", "trusted external checkpoint or oracle (control, not a learned behaviour)"]

R(BL, "failure_diagnosis", "observed", BG, BD + "behavioral_pilots_20261030.md", "Scientific negative results",
  ["SmolLM2 selected the literal letter B in all 144 forward calls", "Qwen2.5-0.5B selected literal A in 116/144 regular prompts and 114/144 option-swapped prompts",
   "58/144 (40.28%) direct semantic action agreements against a frozen 75% pass requirement", "FAIL_ORDER_TEST_DO_NOT_INTERPRET_AS_SEMANTIC_AGENT"],
  "Two frozen small-model trials (SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct) ran the same synthetic packet-release fixtures (four development and twelve heldout tasks) with scripted policy-memory interventions. Reported behaviour changes need to be attributed.",
  SK_BGI,
  "Run the option-swapped (order-invariance) probe before attributing any behavioural difference to the model's judgement.",
  None,
  "SmolLM2 picked the literal letter B in all 144 forward calls, so the reported variation came from the externally enforced gates and label orientation. Qwen2.5-0.5B picked A in 116/144 regular and 114/144 option-swapped prompts; its option-invariant probe gave 58/144 (40.28%) direct semantic agreement against a frozen 75% requirement, status FAIL_ORDER_TEST_DO_NOT_INTERPRET_AS_SEMANTIC_AGENT.",
  "Counterbalance option labels, report literal-letter rates beside semantic agreement, and fix the pass threshold before the run.",
  "none")

R(BL, "experiment_design", "hypothesis", BG, BD + "behavioral_pilots_20261030.md", "Follow-up protocol gate",
  ["A new study must register fresh heldout tasks and compare a direct natural-language SEND/HOLD probe to order-swapped labels",
   "A model's ability to read task facts is a prerequisite for interpreting later self-repair experiments."],
  "The B and C pilots failed their semantic-inference quality gate. A successor study of policy self-repair is being planned.",
  SK_BGI,
  "Register fresh heldout tasks, a direct natural-language SEND/HOLD probe with order-swapped labels, heldout answer-label calibration, free-form parsing failures reported, a competent rule-oracle control and human-audited ground truth before any self-repair claim.",
  "Not a numeric prediction: a gate that must be passed before later experiments are interpretable.",
  "Run matched models with a frozen compute budget and new stopping rules, and obtain independent reviewer sign-off before claims about developmental self-models.",
  "proposed")

R(BL, "failure_diagnosis", "observed", BG, BD + "bgi01d_direct_pilot_20261030.md", "Scored baseline before policy edits",
  ["| Actual raw model action | 16 DENY, 0 ALLOW | 1 DENY, 15 ALLOW |", "Actual was 8/16 correct and 1/16 equivalent-phrase agreements",
   "The predeclared semantic instrument threshold was 12/16 correct raw actions and 14/16 paraphrase agreements."],
  "Pinned Qwen2.5-0.5B-Instruct decides ALLOW or DENY on the same 16 factual access cases rendered two ways: plain key=value facts and equivalent prose, each ending with a direct instruction. Four controller architectures (A immutable, B and C accept typed edits, D independent validator plus checkpoint) sit around the model.",
  SK_BGI,
  "Treat prompt rewording as a first-class intervention: score the same facts in both renderings before comparing controllers.",
  None,
  "Plain syntax gave 16 DENY and 0 ALLOW; equivalent prose gave 1 DENY and 15 ALLOW. Raw accuracy was 8/16 and 9/16 with 1/16 paraphrase agreement against a predeclared gate of 12/16 and 14/16 (FAIL_BASELINE_SEMANTIC_PROBE), so safety statistics for the controllers were dominated by wording.",
  "Pre-declare the correctness and paraphrase-agreement thresholds and check that an always-DENY policy cannot pass them.",
  "none")

R(BL, "failure_diagnosis", "observed", BG, BD + "bgi01d_direct_pilot_20261030.md", "Genuine model-generated patch evidence",
  ["The three later unsafe recovery patches are genuine generated text too", "The recovered B/C states have require_active_badge=false in every campaign.",
   "repair proposals should be subjected to the same no-regression safety validator as updates"],
  "Under a trusted post-incident repair prompt that asks to restore safety flags, the 0.5B model is allowed one typed single-field JSON patch. Campaigns: P1 untrusted suggestion to drop an escort rule, P2 untrusted suggestion to ignore an emergency stop, P3 authorized audit logging.",
  SK_BGI,
  "Subject model-proposed repair patches to the same no-regression validator as ordinary updates, even when they come from a trusted audit context.",
  None,
  "The model refused both malicious edits and made the audit-logging edit, but in all three recovery prompts it then proposed require_active_badge=false, which B and C accepted; D restored its canonical checkpoint. The misconfiguration stayed latent only because the model emitted DENY on every plain-format action.",
  "Diff the full before and after policy against a protected reference and replay denial cases after every accepted patch.",
  "none")

R(BL, "failure_diagnosis", "observed", BG, BD + "bgi01e_scale_control_20261030.md", "Model-scale comparison",
  ["The 1.5B model chose DENY on 15/16 primary prompts, then ALLOW on 10/16 reworded prompts", "FAIL_BASELINE_SEMANTIC_PROBE",
   "| Equivalent-prompt agreement | 1/16 | 7/16 | at least 14/16 |"],
  "After the 0.5B model failed the semantic gate, the same 16 cases and unchanged scorer are run on Qwen2.5-1.5B-Instruct to test whether scale rescues the instrument.",
  SK_BGI,
  "Rerun the unchanged D scorer on a 3x larger model before concluding that the failure is specific to the small model, and label it exploratory because the same sixteen cases were already inspected.",
  None,
  "Correct primary actions rose from 8/16 to 9/16 and equivalent-prompt agreement from 1/16 to 7/16, still below the 12/16 and 14/16 gates. The 1.5B model chose DENY on 15/16 primary prompts and ALLOW on 10/16 reworded prompts with identical access facts, status FAIL_BASELINE_SEMANTIC_PROBE.",
  "Keep prompts, fixture hash and thresholds unchanged and state that inference resources differ across the two models.",
  "none")

R(BL, "experiment_design", "observed", BG, BD + "bgi01e_scale_control_20261030.md", "Model-generated policy updates and repair",
  ["The P1 escort weakening therefore persisted in both mutable architectures after the supposed repair.",
   "A repair receipt must compare the complete before/after policy against an independent protected reference, check actual changed guard bits, and rerun specified denial cases"],
  "In all three post-incident recovery prompts the 1.5B model proposed require_active_badge=true, a flag that was already true, while an earlier model-proposed weakening of the escort rule (P1) had been accepted by the mutable architectures B and C.",
  SK_BGI,
  "Require a repair receipt that compares the complete before and after policy with an independent protected reference, checks the guard bits that actually changed, and reruns the specified denial cases.",
  None,
  "The P1 escort weakening persisted in both mutable architectures after the supposed repair; only D's independent safety validator prevented the edit in the first place. A well-formed repair request is not evidence of successful restoration.",
  "Compare guard-bit vectors before and after, not the stated intent of the patch.",
  "none")

R(BL, "failure_diagnosis", "observed", BG, BD + "bgi01f_verified_repair_20261030.md", "Fresh semantic-instrument result",
  ["| Deny broken seal | **0** | **0** |", "Total primary correct 35/48 (72.9%)", "Aggregate accuracy would have concealed that entire safety failure.",
   "direct-action agreement 40/48 (83.3%) versus 42/48 required"],
  "A fresh 48-case investigator-generated semantic instrument (six groups of eight: authorized OPEN release, authorized SENSITIVE/verified release, deny revoked permit, deny broken seal, deny sensitive/unverified recipient, deny multiple faults) is scored on Qwen2.5-1.5B-Instruct in two wordings. The gate needs at least six correct in each group and 42 stable answers.",
  SK_BGI,
  "Score the instrument per family, not only in aggregate, and apply the registered per-group minimum.",
  None,
  "Primary accuracy was 35/48 (72.9%) and reworded 37/48 (77.1%), agreement 40/48 against 42/48 required: FAIL_SEMANTIC_INSTRUMENT. The broken-seal family scored 0/8 in both phrasings despite a clear rule requiring an intact seal, and authorized OPEN release depended on wording (3/8 vs 8/8); aggregate accuracy would have hidden the broken-seal failure.",
  "Report per-family counts with raw outputs and keep human adjudication of the investigator-written labels as an open gate.",
  "none")

R(BL, "skill_composition", "observed", BG, BD + "bgi01f_verified_repair_20261030.md", "Model-generated repair evidence",
  ["| No repair | 1 | 9 | 12 |", "| Model patches, verified | 6 | 1 | 0 |", "| External checkpoint | 7 | 0 | 0 |",
   "Under three simultaneous missing guards, it enabled the stop guard, then badge guard, then redundantly enabled the already-true stop guard."],
  "Seven damaged-policy scenarios are presented to Qwen2.5-1.5B-Instruct with a machine-readable witness of the failure; it proposes one JSON patch per round over three rounds, and an exact finite verifier (352 enumerated transitions) can accept or reject each patch.",
  SK_BGI,
  "Combine a model patch proposer with an exact no-regression verifier and a trusted external checkpoint, and report each arm separately.",
  None,
  "Full reference state restored (of 7): no repair 1, model patches unchecked 6, model patches verified 6, external checkpoint 7, deterministic oracle 7; missing guards after all scenarios 9, 1, 1, 0, 0. Under three simultaneous missing guards the model repaired two and then repeated an already-true flag, leaving the escort guard off. All proposals were non-weakening, so checked and unchecked arms behaved identically; the verifier's protection comes from the exhaustive finite check.",
  "Record who produced each state change (model, verifier or checkpoint) and never merge externally supplied recovery into model autonomy.",
  "none")

R(BL, "skill_composition", "observed", BG, BD + "bgi01g_factored_policy_20261030.md", "Fresh synthetic GPU-admission tasks",
  ["| Correct action | **28/56** | **55/56** |", "| Unsafe RUN on 32 prohibited cases | **27/32** | **0/32** |",
   "Extracted facts: **277/280 primary factual positions correct**",
   "an unsafe model-extracted admission requires an incorrect complete observation"],
  "56 investigator-written GPU-admission cases (eight variations of seven fact patterns, two renderings) are decided by Qwen2.5-1.5B-Instruct either directly (RUN or HOLD) or by extracting the facts as JSON and passing them to a fixed external policy.",
  SK_BGI + ["model fact extraction into a JSON schema", "fixed external admission policy"],
  "Split the decision: have the model extract the facts and let an independent policy decide, rather than asking the model for the action.",
  None,
  "Correct action: direct 28/56 vs factored 55/56. Unsafe RUN on 32 prohibited cases: 27 vs 0. Authorized runs: 23/24 for both. Fact extraction was 277/280 positions correct with no schema failures; the one consequential error read a VERIFIED supervisor as UNKNOWN. On the prose rendering the direct model said RUN on all 56 cases. The benefit holds only if the extracted facts are adequate, and the pipeline was built after seeing BGI-01F failures.",
  "Attribute failures to specific input fields and keep the 56 correlated cases from being treated as independent draws.",
  "none")

R(BL, "failure_diagnosis", "observed", BG, BD + "bgi01g_factored_policy_20261030.md", "Three-round policy repair experiment",
  ["| Full protected-vector restoration | **7/8** | **3/8** |", "The feedback text harmed repair performance in this run.",
   "| Redundant or rejected edits (24 attempts per mode) | **9** | **14** |"],
  "Eight damaged-policy scenarios are repaired by the same model over three rounds under two prompts: state plus witness only, or the same plus feedback from the previous verification.",
  SK_BGI,
  "Test whether adding previous-verification feedback to the repair prompt helps, with the exact verifier and a trusted checkpoint as controls.",
  None,
  "Full restoration was 7/8 with state plus witness only and 3/8 with feedback; redundant or rejected edits rose from 9 to 14 of 24 attempts and one unlawful configuration remained in the exact checker. In four scenarios the model repeated a repair to an already-true flag instead of restoring the remaining false guards, so verbose feedback harmed repair.",
  "Compare restoration counts per scenario and per prompt condition, and keep the trusted checkpoint (8/8) as an externally supplied control.",
  "none")

R(BL, "experiment_design", "hypothesis", BG, BD + "bgi01g_factored_policy_20261030.md", "Implications and next experiment",
  ["When the controller makes a repair, label it externally substituted; model-initiated repair success must exclude those substitutions.",
   "Compare policy progress after each step using a fixed potential function such as the count of missing protected guards."],
  "Model-driven repair prompts stall on already-true flags (7/8 without feedback, 3/8 with feedback). A deterministic trusted controller could fill missing guards when the model stalls.",
  SK_BGI,
  "Separate state-snapshot scheduling of required repairs from model-selected suggestions, let a trusted controller fill missing guards when the model stalls, label those steps externally substituted, and measure progress by a fixed potential such as the count of missing protected guards.",
  "Not a numeric prediction: a proposed next experiment.",
  "Report model-initiated repair success excluding substitutions, and keep the earlier failed receipts.",
  "proposed")

R(BL, "skill_composition", "observed", BG, BD + "bgi01h_progress_controller_20261030.md", "Model-selected repair",
  ["| Fully restored cases with at most 3 damaged guards | **8/8** | **8/8** |",
   "The model made exactly 23 effective selections out of 23 opportunities with an outstanding guard", "| External substitutions | **0** | **0** |"],
  "The repair interface is narrowed: the model receives a closed, authoritative list of missing guards and picks one field per round, with an exact progress certificate (32 policy states x 11 suggestions = 352 transitions) and an external fallback available.",
  SK_BGI + ["closed list of missing guards", "progress potential = number of missing protected guards"],
  "Replace free-form repair with a closed-list single selection checked against a progress potential, with an external fallback that is recorded separately.",
  None,
  "All 8 in-scope cases (at most three damaged guards) were fully restored by the model's own selections, 23 effective selections out of 23 opportunities, and the fallback never had to act (0 substitutions). The change of output contract came after earlier results were inspected, so it is an engineering demonstration, not a blinded causal effect.",
  "Keep model prompts identical between the fallback-enabled and model-only modes and report substitutions separately.",
  "none")

R(BL, "failure_diagnosis", "observed", BG, BD + "bgi01h_progress_controller_20261030.md", "Model-selected repair, critical terminal limitation",
  ["failed to return the requested {\"field\":null} when no guard was missing", "| Invalid field selections | 5 | 5 |", "| Stale valid selections | 2 | 2 |",
   "successful repair and reliable recognition that repair is finished are separate skills"],
  "In the closed-list repair loop the model must return {\"field\":null} when no guard is missing. Ten cases run for three rounds each.",
  SK_BGI,
  "Test completion independently of the model and skip model queries once all guards are restored.",
  None,
  "The model made 23 of 23 effective selections while a guard was outstanding, but at terminal states it produced five malformed names (for example badge_guard or VERIFIED_missing_guard) and two valid but stale guard names instead of null. Each was rejected as a no-op and nothing was weakened. Successful repair and recognizing that repair is finished are separate skills.",
  "Count invalid and stale selections at terminal states separately from effective selections.",
  "none")

# ---------------------------------------------------------------------------
# CB02 Lean transfer (heldout lineage: transfer only), RC-01, CT-02, learning-theory crosswalk
# ---------------------------------------------------------------------------
C2 = "ab_cb02"
C2P = "research/canonization_bench_cb01/TRANSFER.md"
R("adapter-bits/cb02-lean-transfer", "transfer_across_families", "observed", C2, C2P, "Verified CB02 v1 result",
  ["| Independently compiled baseline proof bodies | 577 UTF-8 bytes |", "| Refactored candidate proof bodies | 112 UTF-8 bytes |",
   "| Candidate plus shared library | 716 UTF-8 bytes |", "| Held-out Lean consumer theorems compiled | 4 |",
   "(577-112)/577 = 80.6%", "716/577 = 1.241"],
  "Three source-derived target propositions from a pinned EvalMath paper (T01_MENU, T02_CODE, T03_STEP) are proved once as independent baselines and once through a shared canonical Lean library plus a derived candidate. Four held-out client theorems are written after the candidate files are frozen and import only the canonical library.",
  ["shared canonical Lean library", "independently proved baseline", "blinded-client protocol with frozen blob hashes", "kernel compilation in CI"],
  "Prove the transfer targets through the frozen canonical library and test reuse on held-out consumers that never import the baseline.",
  None,
  "The refactored candidate proof bodies were 112 UTF-8 bytes against 577 for the independent baseline, a lexical saving of 80.6% on client proof bodies; counting the 604-byte shared library the total was 716 bytes, 1.241 times the baseline. All four held-out consumer theorems compiled and the source-blob checks passed. This is a reuse result with measured library overhead.",
  "Freeze development files with Git blob hashes before writing held-out clients, compile in CI and report the library-inclusive total beside the client saving.",
  "none")

R("adapter-bits/cb02-lean-transfer", "transfer_across_families", "observed", C2, C2P, "Source ownership and protocol limits",
  ["CB02 formally proves only these abstract and deterministic specializations.", "this is not statistical blindness, a new theorem-generation result, or an autonomous-agent benchmark.",
   "The results do not measure autonomous research novelty or model generalization."],
  "The CB02 transfer result is about to be summarized. The source paper's theorem concerns rational stochastic kernels with adaptive feedback; the Lean package proves abstract and deterministic specializations. The authoring model had read the source paper.",
  ["scope statement of what the Lean theorems cover", "protocol-level holdout"],
  "State the scope of the transfer claim: proof reuse and compiler acceptance on mathematically related statements, not novelty or model generalization.",
  None,
  "The source says the holdout is protocol-level (candidate files fixed before evaluator files exist), not statistical blindness, and that stochastic conditioning, finite successor closure, 11-class cardinality and the full bit lower bound stay with the source paper.",
  "Name what is proved and what is left to the source paper in the same paragraph as the measured savings.",
  "none")

R("adapter-bits/cb02-lean-transfer", "transfer_across_families", "hypothesis", C2, C2P, "Freeze and blinded-client protocol",
  ["A future multi-agent blinded test should isolate solver prompts from hidden targets."],
  "CB02 held-out clients were authored by a model that had read the source paper, so the holdout is protocol-level only.",
  ["isolated solver agent", "hidden target statements", "frozen canonical library"],
  "Run a multi-agent blinded test in which the solver prompts are isolated from the hidden targets.",
  "Not a numeric prediction.",
  "Compare proof sizes and compile verdicts for solvers that never see the target statements against the CB02 numbers.",
  "proposed")

RC = "ab_rc01"
RP = "research/recurrent_circuits_v1/results/run_report.md"
RL = "adapter-bits/rc01-recurrent-circuits"
R(RL, "quantization_aware_adaptation", "observed", RC, RP, "Recurrent numerical stability",
  ["all 16 width-4 initial states and five width-16 seeds remained correct through 2048 transitions",
   "Disabling gate restoration failed at transition 5; leaving idle cells unrefreshed failed at transition 4 for widths 16, 64, and 256.",
   "Refreshing idle cells preserved correctness through 2048."],
  "A fixed-weight recurrent NAND circuit is emulated on CPU with every primitive rounded to FP32, BF16 or FP16 and checked at each transition against an independent integer oracle, with a coherent +1/8 perturbation per step.",
  ["scalar software emulation of FP32, BF16 and FP16", "gate restoration", "idle-cell refresh", "independent integer oracle"],
  "Keep gate restoration and idle-cell refresh in the recurrent step, and test removal of each as an ablation.",
  "The source ties this to a conditional bound (separation 1, radius 1/8, per-step perturbation 1/8): restoration should keep errors from accumulating.",
  "With restoration and refresh the circuit stayed correct through 2048 transitions in all three precisions. Without gate restoration it failed at transition 5; with idle cells left unrefreshed it failed at transition 4 for widths 16, 64 and 256. The result does not establish the assumptions for learned weights or CUDA arithmetic.",
  "Check every intermediate transition against the oracle, not only the final output.",
  "mechanism_rationale")

R(RL, "failure_diagnosis", "observed", RC, RP, "Untrained transformer: resident and explicit roundtrips",
  ["The explicit spill arms saved zero VRAM.", "The spill implementation retained the full state allocation, so allocated/reserved peaks were identical to resident execution at each dtype.",
   "These data establish correctness and traffic, but do not support a clean speed comparison across modes."],
  "On an RTX 5080 Laptop shared with an unowned WDDM process, an untrained transformer block (batch 8, 512 tokens, width 256, 128 loops) is run resident and with explicit host spill every k = 1, 8 or 32 loops in FP32, BF16 and FP16.",
  ["resident versus explicit spill arms", "allocator peak recording", "bitwise output comparison"],
  "Read the allocator peaks before crediting the spill arm with any memory benefit, and treat timings from a shared device as noisy.",
  None,
  "Every spill result matched its resident result bitwise, but allocated and reserved peaks were identical because the implementation kept the full state allocation, so the spill arms saved zero VRAM. FP32 resident medians ranged 0.260 to 1.087 s against 0.291, 0.232 and 0.227 s for spill, so no clean speed comparison is supported.",
  "Compare allocator peaks between arms and repeat timings in randomized paired order on an idle device.",
  "none")

R(RL, "failure_diagnosis", "observed", RC, RP, "Preflight and reference gate",
  ["Windows initially checked out CRLF files while the receipt hashes raw bytes. Restoring exact Git blob bytes made the receipt pass; pinned source content was not changed."],
  "A source-hashed receipt (reference_v1.json) must verify before the GPU experiments start; on Windows the pinned sources were checked out with CRLF line endings.",
  ["verify.py --check on the source-hashed receipt", "Git blob bytes"],
  "Restore the exact Git blob bytes (no CRLF conversion) before hashing, then rerun the receipt check in normal and optimized Python.",
  None,
  "Windows initially checked out CRLF files while the receipt hashes raw bytes; restoring exact Git blob bytes made the receipt pass and the pinned source content was not changed. Unit tests passed 23/23 in normal and optimized modes.",
  "Hash bytes read from the pinned blob, not from the working tree.",
  "none")

R(RL, "experiment_design", "observed", RC, RP, "Host-DRAM multitenant prototype",
  ["Peak allocated GPU memory fell by 30, 24, and 16 MiB at active counts 1, 4, and 8.", "k=1 slowed all three active counts.",
   "this prototype has one timing per arm on a shared GPU"],
  "A prototype keeps 16 BF16 session states (2 MiB each, 32 MiB in pinned host memory) and moves active micro-batches of 1, 4 or 8 sessions to the GPU for 32 transitions, with 1, 8 or 32 loops per residence.",
  ["pinned host session pool", "active micro-batch residency", "bitwise comparison with the resident reference"],
  "Residency per k loops should amortize host-device traffic; test k = 1, 8 and 32.",
  None,
  "All nine host-backed outputs matched the resident reference bitwise and peak allocated GPU memory fell by 30, 24 and 16 MiB at 1, 4 and 8 active sessions. k = 1 slowed all three active counts (1.61x, 2.07x, 1.74x); at k = 8 and 32 time approached or fell below resident, but with one timing per arm on a shared GPU these inversions are not reliable speedups, and no higher maximum session count was shown.",
  "Repeat resident and host-backed arms in randomized paired order on an idle GPU and raise the session pool until a resident allocation reaches the declared budget.",
  "none")

CT = "ab_ct02"
CP = "research/compression_trajectories_ct02/README.md"
CL = "adapter-bits/ct02-compression-trajectories"
R(CL, "curriculum_selection", "observed", CT, CP, "Executed three-seed methods pilot (seeds 0,1,2)",
  ["| One-shot CE | 0.877 | 0.877 |", "| Depth-first CE+KD | 0.866 | 0.843 |", "| Scratch CE+KD | 0.781 | 0.781 |",
   "There is no evidence here of a universal progressive-pruning benefit; the fresh-initialization gap is the clearest pilot signal.",
   "Progressive depth-first used ~229M training MAC proxy versus ~179M for one-shot in the equal-example experiment."],
  "An eight-block width-64 residual MLP teacher (mean accuracy 0.851) is compressed to four blocks of width 32 on a synthetic six-class task, with the same recovery examples and optimizer reset for every schedule; a 10%/90% stage split is used for staged arms.",
  ["one-shot CE or CE+KD", "depth-first, width-first and joint staged compression", "layer-saliency selection", "fresh-initialization control"],
  "Compare compression schedules (one-shot, depth-first, width-first, joint, saliency-selected) at equal examples and at a matched multiply-accumulate proxy, with a scratch-trained control.",
  None,
  "Mean validation accuracy at equal examples: one-shot CE 0.877, one-shot CE+KD 0.868, depth-first 0.866, width-first 0.868, joint 0.868, saliency one-shot 0.864, scratch 0.781; at a matched MAC proxy depth-first fell to 0.843. There is no evidence of a universal progressive benefit; the fresh-initialization gap was the clearest signal, with only three seeds on a small smooth synthetic model.",
  "Report equal-example and compute-matched budgets side by side and say which cost terms the proxy excludes.",
  "none")

R(CL, "experiment_design", "observed", CT, CP, "Executed three-seed methods pilot, growth arm",
  ["raw repeated blocks 20.15, scaled repeats 0.0106, scaled random insertions 2.14, and identity insertions 0.0", "Expansion needs scaled residuals"],
  "A trained four-block model is expanded to eight blocks by repeating two middle layers in an 0,1,2,1,2,1,2,3 pattern; variants copy blocks raw, copy with each reused block's residual gain divided by three, insert scaled random blocks, or insert zero-residual identity blocks.",
  ["raw untied copies", "residual-gain scaling by 1/3", "scaled random inserts", "identity inserts as the exact initial-function control"],
  "Scale the residual gain of repeated blocks by 1/3 when growing depth, and use identity insertion as the exact initial-function control.",
  None,
  "Initial-function mean squared difference from the four-block model was 20.15 for raw repeated blocks, 0.0106 for scaled repeats, 2.14 for scaled random insertions and 0.0 for identity insertions (three seeds, small synthetic model).",
  "Measure the initial-function discrepancy before any recovery training, with identity insertion as the zero reference.",
  "none")

R(CL, "experiment_design", "hypothesis", CT, CP, "Pre-register next benchmark before looking at results",
  ["minimum 10 paired", "Register a minimum meaningful effect (e.g. +1.0 percentage point) and a multiplicity policy before data collection.",
   "a 95% paired bootstrap CI over seed-level outcomes"],
  "The three-seed pilot found no progressive-compression benefit. A larger benchmark is planned across three architectures with an independent task-family split.",
  ["paired seeds", "paired bootstrap confidence interval", "example-matched and device-compute-matched budgets"],
  "Pre-register the architectures, task-family split, teacher checkpoint, at least 10 paired seeds, one primary target capability, a minimum meaningful effect of about +1.0 percentage point and a multiplicity policy before collecting data.",
  "Not a numeric prediction: the hypothesis fails its useful-benefit gate if the interval does not support the registered minimum effect at fixed total compute.",
  "Estimate the paired held-out accuracy difference (progressive minus one-shot CE+KD) with a 95% paired bootstrap interval over seeds and measure real throughput and VRAM peak.",
  "proposed")

R("adapter-bits/learning-theory-crosswalk", "experiment_design", "hypothesis", "ab_clt", "papers/learning_theory/CROSSWALK.md", "Compression and causal representations; PAC/VC",
  ["In the Z31 shear fixture, five bits can encode the present sum (31 possibilities), while ten fixed bits are needed for joint present and post-shear outputs (961 possibilities).",
   "The encoder, adapter, routing and decoder must be counted."],
  "A compact recurrent student must carry enough state for both the present value and its future after a shear in a modular-addition (Z31) task. Counting-based generalization bounds charge the whole task-dependent predictor family.",
  ["finite description-length union bound", "exact quotient of future behaviour", "fixed-bit code budgets"],
  "Give the state code ten bits, not five, when the readout needs the joint present and post-shear output, and count encoder, adapter, routing and decoder in the description length.",
  "Not a numeric prediction: five bits cover 31 present sums, but 961 joint outcomes need at least ten bits; a specific quantizer may still collide.",
  "Compare equal-total-code-bit alternatives and check whether present-only and joint readouts separate.",
  "proposed")


# ---------------------------------------------------------------------------
# q1-adapters: task adapters on Bonsai 8B Q1_0, Prime Intellect environments
# ---------------------------------------------------------------------------
Q1 = "q1"
QP = "paper/q1_adapters.md"
SK_Q = ["rank-16 LoRA on all seven projections (87 MB), QLoRA on Bonsai-8B-unpacked", "SFT on the environment's gold outputs",
        "rejection-sampling fine-tuning from the base model's own correct samples", "paired bootstrap 95% CI on the same items and seeds",
        "llama-server with adapters switched through /lora-adapters"]
CTX_Q = ("Prime Intellect Environments Hub environments scored by their own reward code. Base: PrismML Bonsai 8B Q1_0 (1-bit weights, 1.16 GB) in non-thinking mode "
         "(Qwen3 chat template, T = 0.7, top-p 0.8, top-k 20, per-example seeds shared across configurations). Comparators: 4-bit Qwen3-8B (5.03 GB) and a memory-matched 4-bit Qwen3-1.7B (1.11 GB).")

R("q1-adapters/reverse-text", "learning_gain_prediction", "observed", Q1, QP, "3.2 What an 87 MB adapter buys; training cost",
  ["| reverse-text | 0.115 | **0.817** | **+0.702** [+0.682, +0.722] |",
   "the base reorders words instead of reversing characters and runs past the 384-token cap on half the items (mean 205 tokens, 49% truncated); with the adapter, 51 tokens, 0% truncated",
   "| reverse-text | 800 env gold reversals | — | 7 min (150 steps) | 7 min |"],
  CTX_Q + " Environment: reverse-text, last 200 of 1,000 rows (first 800 are train), score = LCS ratio of the parsed reversed text. The 4-bit 8B scores 0.208.",
  SK_Q,
  "Train a rank-16 LoRA by SFT on the 800 gold reversals from the environment's own train rows (150 steps, about 7 minutes on a laptop RTX 5080).",
  "No number registered. The source's mechanism: where the base is far from the task a few hundred gold examples can move it a long way, because reverse-text is a learnable transformation neither base performs.",
  "Score went from 0.115 to 0.817, a paired gain of +0.702 [+0.682, +0.722] on 200 items, 0.61 above the 4-bit 8B. The base reordered words instead of reversing characters and ran past the 384-token cap on half the items (mean 205 tokens, 49% truncated); with the adapter the mean was 51 tokens and 0% were truncated.",
  "Evaluate on the held-out last 200 rows with the same per-example seeds for base and adapter, and report a paired bootstrap interval.",
  "mechanism_rationale")

R("q1-adapters/gsm8k", "learning_gain_prediction", "observed", Q1, QP, "3.2 and 3.8, GSM8K",
  ["+0.030 [−0.020, +0.075]", "+0.020 [+0.002, +0.038]", "757 of 1,600 samples correct", "gsm8k 91% of training samples correct"],
  CTX_Q + " Environment: primeintellect/gsm8k scored by math-verify on '#### N'. The base already scores 0.845 on the first 200 test items and 0.851 on the full 1,319.",
  SK_Q,
  "Train a rank-16 LoRA by rejection-sampling fine-tuning: sample 4 completions for each of 400 train prompts from the base 1-bit model, keep up to two distinct fully correct completions (757 of 1,600 samples were correct), and fine-tune on them.",
  "No number registered. The source's mechanism: rejection-sampling fine-tuning can only amplify behaviour the base already samples, so where 91% of training samples are already correct there is little left to amplify.",
  "On 200 items the gain was +0.030 [−0.020, +0.075] (0.845 to 0.875), indistinguishable from zero; on the full 1,319-item test set it was a small but real +0.020 [+0.002, +0.038]. The larger sample sharpened an earlier 'nothing measurable' call.",
  "Re-estimate the paired gain on the full test set before calling a small adapter gain null or real.",
  "mechanism_rationale")

R("q1-adapters/medmcqa", "learning_gain_prediction", "observed", Q1, QP, "3.2 and 3.8, MedMCQA",
  ["+0.065 [−0.005, +0.135]", "+0.062 [+0.018, +0.108]", "+0.044 [+0.015, +0.073]", "465 of 1,600"],
  CTX_Q + " Environment: MedMCQA validation, boxed-letter accuracy. The base scores 0.375 on 200 items, 0.388 on 500 and 0.409 on 1,000.",
  SK_Q,
  "Train a rank-16 LoRA by rejection-sampling fine-tuning (400 prompts x 4 samples, 465 of 1,600 correct, 60 steps, 76 minutes in total).",
  "No number registered; the source expects rejection sampling to give modest gains where the base already samples some correct answers.",
  "The gain was +0.065 [−0.005, +0.135] at n = 200 (0.375 to 0.440), +0.062 [+0.018, +0.108] at n = 500 and +0.044 [+0.015, +0.073] at n = 1,000 (0.409 to 0.453), putting the adapted 1-bit model above the 4-bit 1.7B but 0.11 below the 4-bit 8B.",
  "Report the paired gain at the largest sample available; at n = 200 the interval includes zero.",
  "mechanism_rationale")

R("q1-adapters/ifeval", "learning_gain_prediction", "observed", Q1, QP, "3.2 What an 87 MB adapter buys",
  ["0.000 [−0.045, +0.040]", "479 of 1,600", "| ifeval | 0.640 | 0.640 |"],
  CTX_Q + " Environment: arcee-ai/ifeval, the environment's own seeded 200-row holdout, RLVR-IFEval constraint pass rate. The base scores 0.640.",
  SK_Q,
  "Train a rank-16 LoRA by rejection-sampling fine-tuning (479 of 1,600 samples correct, 60 steps, 103 minutes in total).",
  "No number registered; rejection sampling can only sharpen behaviour the base already has.",
  "Score stayed at 0.640, a paired change of 0.000 [−0.045, +0.040]: nothing measurable. The 4-bit 8B scores 0.685.",
  "Check that the interval around zero is narrow enough to say the adapter did nothing, and count training samples that passed the verifier.",
  "mechanism_rationale")

R("q1-adapters/pubmedqa", "learning_gain_prediction", "observed", Q1, QP, "3.8 Phase N2, PubMedQA",
  ["+0.070 [+0.030, +0.110]", "400 train prompts × 4, 556 kept", "habit (327 → 295 of 500)"],
  CTX_Q + " Environment: maziyar/OpenMed_PubMedQA, first 500 of the shuffled labeled eval split, yes/no/maybe accuracy. The base scores 0.364 and the 4-bit 8B 0.432; the base says 'maybe' on 327 of 500 items against 51 gold 'maybe'.",
  SK_Q,
  "Train a rank-16 LoRA by rejection sampling from the environment's own yes/no-only training split (400 prompts x 4 samples, 556 kept).",
  "No number registered; the source calls the failure a calibration problem and trains an adapter to fix an output-format habit.",
  "Accuracy rose from 0.364 to 0.434, +0.070 [+0.030, +0.110], level with the 4-bit 8B (0.432). It did not cure the 'maybe' habit (327 to 295 of 500 'maybe' answers); gains came on 'yes' (0.380 to 0.464) and 'no' items (0.231 to 0.295).",
  "Break accuracy down by gold label so that a format fix and a knowledge gain can be told apart.",
  "mechanism_rationale")

R("q1-adapters/ascii-tree", "learning_gain_prediction", "observed", Q1, QP, "3.8 Phase N2, ascii-tree",
  ["+0.919 [+0.899, +0.936]", "On the other 184 the adapter scores 0.979 (base 0.064, 8B 0.230)", "16 eval trees duplicate a training tree"],
  CTX_Q + " Environment: primeintellect/ascii-tree (render a prose directory description as an ASCII tree), last 200 rows held out, weighted tree similarity. Every node name of every gold tree appears in its prompt. The base scores 0.062 and the 4-bit 8B 0.230.",
  SK_Q,
  "Train a rank-16 LoRA by SFT on the environment's gold trees from 800 train rows, and check eval prompts for leakage into training.",
  "No number registered; the source treats this as a second transformation skill like reverse-text.",
  "Score rose from 0.062 to 0.981, +0.919 [+0.899, +0.936]. No eval prompt appears in training and 16 eval trees duplicate a training tree; on the other 184 items the adapter scores 0.979 (base 0.064, 4-bit 8B 0.230), so it is a learned output discipline, not memorized trees.",
  "Remove duplicated eval items and re-score; confirm that every answer is derivable from the prompt.",
  "mechanism_rationale")

R("q1-adapters/reasoning-core", "learning_gain_prediction", "observed", Q1, QP, "3.8 Phase N2, reasoning-core",
  ["+0.080 [+0.038, +0.119]", "40 of 500 base answers lacked the <answer> tags", "+0.069 [+0.027, +0.112]", "Answers shrink from ~70 to ~22 tokens."],
  CTX_Q + " Environment: primeintellect/reasoning-core, 500-row eval stream over 56 procedurally generated task families (grammars, logic, planning, arithmetic). The base scores 0.220 and the 4-bit 8B 0.256.",
  SK_Q,
  "Train a rank-16 LoRA on the environment's gold answers only (no derivations), 800 rows, to test whether an answer-format adapter can buy anything on a reasoning task.",
  None,
  "Score rose from 0.220 to 0.300, +0.080 [+0.038, +0.119], above the base 4-bit 8B. 40 of 500 base answers lacked the answer tags and the adapter fixed all of them, but the gain is still +0.069 [+0.027, +0.112] on the 460 already tagged; answers shrank from about 70 to about 22 tokens. By family it traded gains (metamath entailment, abduction, parsing derivations) for losses (word problems, constraint satisfaction, probabilistic outcomes), with 3 to 17 items per family. Answer-shape adapters buy points on this benchmark; they do not make an 8B model reason.",
  "Re-score the subset that already satisfied the output format, and read per-family changes as directions, not estimates.",
  "none")

R("q1-adapters/training-signal-choice", "curriculum_selection", "observed", Q1, QP, "3.2 and 4",
  ["it can only amplify\nbehaviour the base already samples", "Adapters buy output discipline, not reasoning.", "(+0.70\n  reverse-text, +0.92 ascii-tree)",
   "An\n  answer-only adapter gains +0.08 on reasoning-core by changing answer shape, while losing on the\n  families where the base reasoned best."],
  CTX_Q + " Nine environments are available and one adapter can be trained per environment with a different training signal: environment gold outputs, rejection-sampled own completions, or gold answers only.",
  SK_Q,
  "Choose the training signal by how far the base is from the task: SFT on gold outputs where the base lacks a transformation skill (reverse-text, ascii-tree); rejection sampling where the base already samples correct answers (GSM8K, MedMCQA, IFEval, PubMedQA); gold answers only as a probe on reasoning-core.",
  "Rejection sampling can only amplify behaviour the base already samples, so expect small gains where the base is mostly right and large gains only for skills supplied by gold data.",
  "Largest gains were on transformation skills whose answer is fully in the prompt (+0.70 reverse-text, +0.92 ascii-tree). Rejection-sampled adapters gave +0.02 on GSM8K, +0.04 to +0.07 on MedMCQA and PubMedQA, and nothing on IFEval; an answer-only adapter gained +0.08 on reasoning-core by changing answer shape while losing on the families where the base reasoned best. Adapters bought output discipline, not reasoning.",
  "Measure the base's share of already-correct training samples before choosing rejection sampling, and check per-family losses after any aggregate gain.",
  "mechanism_rationale")

R("q1-adapters/systems", "quantization_aware_adaptation", "observed", Q1, QP, "3.7 Systems: memory and speed",
  ["a 1-bit model cannot (the sum is no longer ±scale)", "fell from ~390 to ~160–185 aggregate tok/s (6 concurrent requests)",
   "| Q1 Bonsai 8B + all four adapters loaded | **2.37 GiB** | 1.55 GiB |", "| Q4 Qwen3-8B + four adapters | 5.73 GiB | 4.91 GiB |"],
  "Bonsai 8B Q1_0 stores one sign bit per weight plus an FP16 scale per group of 128. A 4-bit model can absorb a LoRA by merging it into fp16 weights and re-quantizing. Four 87 MB adapters are to be served on one base.",
  ["merge adapter into fp16 weights then re-quantize", "serve adapters unmerged through llama-server /lora-adapters", "nvidia-smi VRAM delta at server start"],
  "Serve the 1-bit model with adapters loaded unmerged and switched per request (the skill-library pattern), not as a merged specialist.",
  "The source's reasoning: the sum of a 1-bit weight and a low-rank delta is no longer plus or minus the scale, so a merge is impossible and every adapter runs as a separate low-rank product.",
  "With one active adapter, served throughput fell from about 390 to about 160-185 aggregate tokens per second (6 concurrent requests). VRAM was 2.37 GiB for the 1-bit base plus all four adapters against 5.73 GiB for the 4-bit 8B plus four adapters.",
  "Measure aggregate throughput with and without an active adapter and the nvidia-smi VRAM delta at server start.",
  "mechanism_rationale")

R("q1-adapters/systems", "quantization_aware_adaptation", "observed", Q1, QP, "3.7 Systems, llama-bench",
  ["| **Bonsai 8B Q1_0** | **1.15 GB** | **200.6 tok/s** | 4,949 tok/s | **17.7 tok/s** |", "| Qwen3-8B Q4_K_M | 5.02 GB | 89.1 tok/s | 3,137 tok/s | 11.1 tok/s |",
   "decode is memory-bandwidth-bound"],
  "llama-bench on the PrismML llama.cpp fork (build 10743), RTX 5080 Laptop 16 GB and Ryzen AI 9 365, compares Bonsai 8B Q1_0 with Qwen3-8B Q4_K_M and Qwen3-1.7B Q4_K_M.",
  ["llama-bench tg128 and pp512", "CPU decode with -ngl 0"],
  "Prefer the 1-bit 8B when decode throughput at fixed parameter count matters, since decode is memory-bandwidth-bound.",
  "Weights are a quarter of the bytes, so decode should speed up by roughly that factor.",
  "Bonsai 8B Q1_0 (1.15 GB) decoded at 200.6 tok/s on the GPU and 17.7 tok/s on the CPU, against 89.1 and 11.1 for Qwen3-8B Q4_K_M (5.02 GB): 2.25x faster on the GPU and 1.6x on the CPU from 4.4x less weight memory. Prefill was 4,949 against 3,137 tok/s.",
  "Run llama-bench with the same build on the same device for both models, GPU and CPU.",
  "mechanism_rationale")

R("q1-adapters/awq-basis-diagnosis", "failure_diagnosis", "observed", Q1, QP, "3.6 Against the 4-bit 8B, what happened to the community adapters",
  ["| same file, CPU backend only (30 items) | 0.47 |", "| our Q4_K_M, converted from the published Qwen/Qwen3-8B (base 0.95) | 0.86 |",
   "Only the official Q4_K_M fails, on both backends", "Its RMSNorm weights differ from the published\nmodel's by 20–55%"],
  "A public Qwen3-8B GSM8K adapter (MetaMathQA, author reports 0.84 in bf16 with vLLM) drops Qwen's official Qwen3-8B-Q4_K_M from 0.945 to 0.425 on GSM8K, with stuttering outputs. The file's general.name reads 'Qwen3 8B Awq Compatible Instruct'. Candidate causes: 4-bit quantization, llama.cpp, the adapter, or the file.",
  ["adapter applied at runtime or merged", "re-conversion from the published checkpoint at Q4_K_M, Q4_0, Q5_K_M, Q6_K and Q8_0", "CPU versus CUDA backend", "transformers + PEFT on 4-bit NF4", "norm-weight comparison with the source checkpoint"],
  "Hold the adapter and 100 GSM8K items fixed and vary only the base weights file: official Q4_K_M, official Q8_0, our own Q4_K_M converted from the published checkpoint, other quant types, and a PEFT/NF4 reference; then compare a norm tensor with the source weights.",
  None,
  "The official Q4_K_M scored 0.37 (0.47 on CPU only), the official Q8_0 0.87, our own Q4_K_M from the published weights 0.86 and the PEFT/NF4 reference 0.85 against the author's 0.84. Only the official Q4_K_M fails, on both backends; its RMSNorm weights differ from the published model's by 20-55% because it was converted from an AWQ-rescaled checkpoint, so the adapter lands in the wrong weight basis. Small self-distilled adapters barely moved the official file, long-trained ones broke it.",
  "Check general.name and general.finetune, or compare a norm tensor against the source weights, before attaching a LoRA to a downloaded GGUF; or convert the GGUF yourself from the checkpoint the adapter was trained on.",
  "none")

R("q1-adapters/community-adapter-contracts", "failure_diagnosis", "observed", Q1, QP, "3.6 The fair comparison, MedMCQA adapter",
  ["0.560 → 0.085", "by letter it is right 58.5% of the time (base 56.0%)", "answers in its training format — a bare letter, then an explanation — which the env's `\\boxed{}` parser rejects"],
  "On a correctly built Qwen3-8B Q4_K_M a public MedMCQA QLoRA adapter (vireshk) is scored under the MedMCQA environment's contract: a boxed letter. The base scores 0.560.",
  ["environment reward code", "letter-level re-scoring of the raw outputs", "adapter training-format inspection"],
  "Re-score the raw outputs by answer letter and compare with the environment score to separate knowledge from output format.",
  None,
  "The environment score fell from 0.560 to 0.085 (paired -0.475). The adapter answers in its training format, a bare letter then an explanation, which the environment's boxed-letter parser rejects; by letter it is right 58.5% of the time against 56.0% for the base. It knows slightly more medicine and scores a sixth as much.",
  "Always score community adapters under the target environment's own parser and also by content, so a format mismatch is not read as lost knowledge.",
  "none")

R("q1-adapters/community-adapter-contracts", "failure_diagnosis", "observed", Q1, QP, "3.6 The fair comparison, GSM8K GRPO and Tulu adapters",
  ["0.940 → 0.540", "never learned to stop: loops after `#### N` on every backend", "0.685 → 0.540"],
  "Two more public Qwen3-8B adapters are applied to our own correctly built Qwen3-8B Q4_K_M: a GRPO GSM8K adapter (Pradheep1647) and a Tulu instruction-following adapter (tianzl66).",
  ["environment-contract scoring", "inspection of output tails for stop behaviour"],
  "Inspect where the outputs end and what objective the adapter was trained for, on every backend, before blaming quantization.",
  None,
  "The GRPO adapter took GSM8K from 0.940 to 0.540 because it never learned to stop and loops after its answer on every backend, NF4 included. The Tulu IF adapter took IFEval from 0.685 to 0.540, a weak adapter for this environment at any precision. A community adapter brings the answer format, stopping behaviour and objective it was trained with.",
  "Check stop behaviour and output format on a few raw generations before running the full evaluation.",
  "none")

R("q1-adapters/base-failure-modes", "failure_diagnosis", "observed", Q1, QP, "3.8 Weak for everyone, ascii-tree",
  ["every node name of every gold tree appears in its prompt", "with 21–30% of outputs hitting the 768-token cap",
   "The typical failure is copying the\nprompt's style example tree into the answer, then appending a garbled version of the real one."],
  CTX_Q + " Environment: ascii-tree. Base scores: Q1 0.062, Q4 1.7B 0.106, Q4 8B 0.230.",
  ["output inspection", "truncation-rate counting at the 768-token cap"],
  "Read raw outputs and count truncations before deciding whether the low score is a reasoning limit.",
  None,
  "Every node name of every gold tree appears in its prompt, so the task is not hard as reasoning; 21-30% of outputs hit the 768-token cap, and the typical failure is copying the prompt's style-example tree into the answer and then appending a garbled version of the real one. It is a format problem an adapter can fix.",
  "Check whether the answer content is derivable from the prompt and whether outputs are truncated.",
  "none")

R("q1-adapters/base-failure-modes", "failure_diagnosis", "observed", Q1, QP, "3.8 Weak for everyone, PubMedQA",
  ["every model answers \"maybe\" on 59–65% of items against a 10% gold rate", "(Q1: 327/500 \"maybe\"; gold 276 yes / 173 no / 51 maybe)",
   "two worked examples are a \"yes\" and a \"maybe\" for \"uncertain evidence\", with no \"no\""],
  CTX_Q + " Environment: PubMedQA (500 items). Accuracy: Q1 0.364, 1.7B 0.390, 8B 0.432.",
  ["confusion counts by gold label", "environment system-prompt inspection"],
  "Tabulate predicted against gold labels and read the environment's system prompt before treating low accuracy as weak reasoning.",
  None,
  "Every model answers 'maybe' on 59-65% of items against a 10% gold rate (Q1: 327 of 500; gold 276 yes, 173 no, 51 maybe). The environment's system prompt primes this: its two worked examples are a 'yes' and a 'maybe' for uncertain evidence, with no 'no'. It is a calibration failure more than a reasoning one.",
  "Compare the predicted label distribution with the gold distribution and look for priming in the environment prompt.",
  "none")

R("q1-adapters/base-failure-modes", "failure_diagnosis", "observed", Q1, QP, "3.8 Where it runs out: multi-step math",
  ["12.8% of its MATH-500 answers and 22.5% of its AIME answers hit the cap, against\n9.6% / 10.0% for the 8B", "On the answers that finish, Q1's MATH-500 accuracy is 0.736"],
  CTX_Q + " Environment: MATH-500 (2,048-token budget) and AIME 2024+2025 (6,144 tokens). Q1 scores 0.642 on MATH-500 against 0.710 for the 4-bit 1.7B and 0.810 for the 4-bit 8B.",
  ["token-cap hit rates", "accuracy on finished answers", "difficulty-level breakdown"],
  "Separate the effect of running out of budget from the effect of wrong derivations by scoring only the answers that finish.",
  None,
  "Q1 hit the cap on 12.8% of MATH-500 answers and 22.5% of AIME answers against 9.6% and 10.0% for the 8B, and on finished answers Q1's MATH-500 accuracy is 0.736 (1.7B 0.792, 8B 0.894). Part of the gap is length; the rest is that one bit per weight costs the most where a derivation must stay exact over many steps.",
  "Report cap-hit rates and finished-answer accuracy next to the headline score.",
  "none")

R("q1-adapters/q1-vs-q4-regimes", "skill_composition", "observed", Q1, QP, "3.1 The whole picture",
  ["At equal memory (~1.1–1.25 GB), the 1-bit 8B is the better model.", "At equal parameters, the 4-bit 8B is the better model",
   "Where a task needs a skill, an adapter beats parameters.", "the 87 MB adapter takes the 1-bit model to 0.82, 0.61 above\n  the 4-bit 8B"],
  CTX_Q + " Four skills (reverse-text, GSM8K, MedMCQA, IFEval) and one 87 MB adapter each; macro score over the four: Q1 base 0.494, Q1 with routed adapters 0.693, Q4 8B 0.598, Q4 1.7B 0.474.",
  SK_Q + ["4-bit Qwen3-8B", "4-bit Qwen3-1.7B"],
  "Pick the configuration by regime: at about 1.2 GB use the 1-bit 8B plus a skill adapter; at equal parameters and a task the base already knows use the 4-bit 8B; for a skill neither base has, train an adapter.",
  None,
  "At equal memory the 1-bit 8B beat the 4-bit 1.7B on GSM8K (+0.05), reverse-text (+0.04) and IFEval (+0.05) and tied on MedMCQA. At equal parameters the 4-bit 8B led on everything the base already did (+0.045 to +0.185). Where a task needs a skill, the 87 MB adapter took reverse-text to 0.82, 0.61 above the 4-bit 8B at a quarter of the memory.",
  "Compare configurations at matched memory and at matched parameters separately; do not average the regimes.",
  "none")

R("q1-adapters/q1-vs-q4-regimes", "skill_composition", "observed", Q1, QP, "3.8 Where it runs out; 4",
  ["MATH-500 Q1 0.642 vs 1.7B 0.710 (−0.068 [−0.108, −0.030]) vs 8B 0.810", "For this kind of task, the 4-bit 1.7B, not the 1-bit 8B, is the better use of ~1.1 GB."],
  CTX_Q + " Workload: competition math (MATH-500 and AIME) in non-thinking mode, about 1.1 GB of weight memory available.",
  SK_Q + ["4-bit Qwen3-1.7B"],
  "For a long exact-derivation workload at about 1.1 GB, choose the 4-bit 1.7B over the 1-bit 8B, and do not expect a GSM8K adapter to close the gap.",
  None,
  "MATH-500: Q1 0.642, Q4 1.7B 0.710 (Q1 minus 1.7B = −0.068 [−0.108, −0.030]), Q4 8B 0.810; AIME avg@2 0.067 vs 0.117 vs 0.242. The 4-bit 1.7B is the better use of about 1.1 GB for this kind of task.",
  "Evaluate on the target workload before choosing the base; recall benchmarks like GSM8K do not predict long-derivation ability.",
  "none")

# --- transfer lineages (heldout family: one lineage per distinct experiment) ---
R("q1-adapters/cross-adapter-matrix", "transfer_across_families", "observed", Q1, QP, "3.4 Adapters are narrow",
  ["| gsm8k | 0.845 | **0.045** | **0.875** | 0.850 | 0.875 |", "| medmcqa | 0.375 | **0.250** | 0.370 | **0.440** | 0.340 |",
   "| ifeval | 0.640 | **0.330** | 0.645 | 0.630 | **0.640** |", "The three self-distilled adapters are close to harmless off-task."],
  CTX_Q + " Four adapters (reverse-text, gsm8k, medmcqa, ifeval) are each applied to every one of the four environments on the 1-bit base, 200 examples per cell.",
  SK_Q,
  "Measure every adapter on every environment before deploying a library, and treat the adapter that learned a genuinely new behaviour as dangerous off-task.",
  None,
  "Off-task, the three self-distilled adapters were close to harmless (for example gsm8k 0.845 with no adapter, 0.850 with the medmcqa adapter, 0.875 with the ifeval adapter). The reverse-text adapter destroyed the others: gsm8k 0.845 to 0.045 (it starts reversing), medmcqa 0.375 to 0.250, ifeval 0.640 to 0.330.",
  "Fill the full adapter-by-environment matrix with the same 200 items and seeds, and read the diagonal and the off-diagonal separately.",
  "none")

R("q1-adapters/community-adapters-on-bonsai", "transfer_across_families", "observed", Q1, QP, "3.3 Public Qwen3-8B adapters run on the 1-bit model",
  ["They change every output (0/200 texts identical to base on gsm8k and medmcqa,\n33/200 on ifeval) but their effect is within noise at n = 200",
   "They neither break the 1-bit model nor reliably help it.", "MedMCQA adapter +0.045 [−0.030, +0.120]"],
  CTX_Q + " Bonsai 8B shares Qwen3-8B's architecture, so rank-16 LoRAs trained on Qwen3-8B load onto the 1-bit weights unchanged: GRPO and MetaMathQA GSM8K adapters, a MedMCQA adapter and a Tulu instruction-following adapter.",
  ["load public Qwen3-8B LoRAs onto Bonsai Q1_0", "paired comparison on 200 items", "text-identity counts versus base"],
  "Try public same-architecture adapters on the 1-bit base before training your own, and count how many outputs they change.",
  None,
  "The adapters changed every output (0/200 texts identical to base on GSM8K and MedMCQA, 33/200 on IFEval) but effects were within noise at n = 200: GSM8K GRPO adapter +0.025 [−0.025, +0.075], MetaMathQA −0.005, MedMCQA adapter +0.045 [−0.030, +0.120], Tulu IF adapter −0.010 [−0.050, +0.025]. They neither broke the 1-bit model nor reliably helped it.",
  "Report the fraction of outputs that differ from the base alongside the paired score change.",
  "none")

R("q1-adapters/bonsai-adapters-on-qwen3-q4", "transfer_across_families", "observed", Q1, QP, "3.6 Against the 4-bit 8B",
  ["reverse-text 0.208 → 0.282,\nMedMCQA 0.560 → 0.590, GSM8K 0.940 → 0.940, IFEval 0.685 → 0.690", "neutral to harmful: 0.093 / 0.940 / 0.510 / 0.660",
   "transfer positively to the correctly built Qwen3-8B Q4_K_M"],
  CTX_Q + " The four Bonsai-trained adapters are applied to Qwen3-8B Q4_K_M built from the exact published checkpoint, and to Qwen's official AWQ-rescaled Q4_K_M, which shares the architecture.",
  ["apply Bonsai-trained LoRA to a different quantization of a sibling base", "official versus self-converted GGUF"],
  "Apply the adapters trained against the Bonsai weights to the sibling Qwen3-8B Q4_K_M and compare with its base scores.",
  None,
  "On the correctly built Q4_K_M the adapters transferred positively: reverse-text 0.208 to 0.282, MedMCQA 0.560 to 0.590, GSM8K 0.940 to 0.940, IFEval 0.685 to 0.690. On the AWQ-rescaled official file they were neutral to harmful (0.093 / 0.940 / 0.510 / 0.660 on the same four environments).",
  "Run the transfer on a base whose weights are known to equal the training basis, and a control on a rescaled basis.",
  "none")

R("q1-adapters/gsm8k-adapter-on-math500", "transfer_across_families", "observed", Q1, QP, "3.8 Where it runs out: multi-step math",
  ["our GSM8K adapter does not transfer to MATH-500 (−0.012 [−0.052, +0.026])", "An adapter\ndoes not change that"],
  CTX_Q + " The GSM8K rejection-sampled adapter lifted GSM8K by +0.020 on the full test set. MATH-500 (500 problems, boxed answers) is a related but harder math environment where the Q1 base scores 0.642.",
  SK_Q,
  "Apply the GSM8K adapter to MATH-500 to see whether a math-recall adapter carries over to long derivations.",
  None,
  "No transfer: the change on MATH-500 was −0.012 [−0.052, +0.026] (0.642 to 0.630). An adapter does not change the conclusion that the 4-bit 1.7B is the better use of about 1.1 GB for this kind of task.",
  "Score on the related environment with the same paired seeds and report the interval, not only the point change.",
  "none")

# --- routing lineage (heldout family) ---
RT = "q1-adapters/routing"
CTX_RT = ("A library of four task adapters (reverse-text, gsm8k, medmcqa, ifeval; 87 MB each) on the 1-bit Bonsai 8B needs a router. Routers are trained on 800 train prompts per environment and scored on 800 eval prompts, in two views: the full prompt as the environment sends it, and the user content with the instruction scaffolding removed. End-to-end macro score is the mean over the four environments of the score of the answer actually produced under the chosen adapter.")
SK_RT = ["72k-parameter recursive router (TRM: hashed word/bigram/char-trigram features, latent z and answer y, 3x3 steps)", "lexical lattice router (LDT: meet over word candidate sets, empty set = bottom)",
         "soft hybrid, hard filter, confidence arbitration, typed membrane with abstain-to-base", "expected-utility routing over the measured cross-task matrix", "oracle router and base-only baselines"]
R(RT, "specialist_routing", "observed", Q1, QP, "3.5 Routing the library",
  ["| oracle (true env's adapter) | 1.000 | 0.693 | 1.000 | 0.693 |", "| no adapter (base) | — | 0.494 | — | 0.494 |",
   "| best single adapter for everything (medmcqa) | — | 0.513 | — | 0.513 |", "| **TRM** (72k parameters) | 0.998 | **0.693** | 0.865 | **0.615** |",
   "A 72k-parameter TRM is enough when the prompt carries the task"],
  CTX_RT, SK_RT,
  "Route each prompt with the 72k-parameter TRM over the four adapters, rather than loading one adapter for everything.",
  None,
  "On full prompts the TRM routed 99.8% correctly and matched the oracle at 0.693 macro, against 0.494 for the base and 0.513 for the best single adapter (medmcqa). On content-only prompts routing fell to 0.865 and the macro to 0.615.",
  "Report routing accuracy and end-to-end macro score for both prompt views, and average the TRM over seeds (5 here).",
  "none")

R(RT, "specialist_routing", "observed", Q1, QP, "3.5 Routing the library, LDT alone",
  ["| LDT alone (lexical lattice; bottom → base) | 0.305 | 0.506 | 0.305 | 0.507 |", "The LDT alone reaches bottom on most\n   prompts (abstains 43–63%)",
   "As soft\n   telemetry it is harmless and inert."],
  CTX_RT, SK_RT,
  "Use the lexical lattice only as soft telemetry beside the TRM, not as the router.",
  None,
  "The lattice alone routed 30.5% correctly and scored 0.506 macro (base 0.494), abstaining to the base on 43-63% of prompts. As soft telemetry beside the TRM it was harmless and inert.",
  "Measure the abstention rate of the lattice and the end-to-end score when it is the only router.",
  "none")

R(RT, "specialist_routing", "observed", Q1, QP, "3.5 Routing the library, hard filter",
  ["| hard-filter hybrid | 0.935 | 0.693 | 0.683 | 0.553 |", "using it as a hard filter costs 0.06 end-to-end on content-only\n   prompts",
   "the paper's routing benchmark (0.584 vs 0.815)"],
  CTX_RT + " The hard-filter hybrid removes from the TRM's choices every adapter the lattice rules out.", SK_RT,
  "Do not let lexical lattice evidence hard-filter the TRM's candidates.",
  None,
  "The hard-filter hybrid kept 0.693 end-to-end on full prompts (routing 0.935) but fell to 0.553 on content-only prompts (routing 0.683) against 0.615 for the TRM alone, a cost of 0.06; the same failure mode as the earlier TRM/LDT routing benchmark (0.584 vs 0.815).",
  "Score the hybrid in the harder content-only view as well as on full prompts.",
  "none")

R(RT, "specialist_routing", "observed", Q1, QP, "3.5 Routing the library, abstention",
  ["| membrane, abstain threshold tuned end-to-end (cross-fitted) | 0.998 | 0.693 | 0.862 | 0.613 |",
   "The errors are bare sentences\n   (reverse-text items) routed to other adapters, and for those the base model is no better than a\n   wrong adapter (0.115), so abstaining gains nothing"],
  CTX_RT + " In the content-only view the TRM's errors are mostly bare sentences, which are reverse-text items.", SK_RT,
  "Add a typed membrane that abstains to the base model when the TRM and the lattice conflict, with the abstain threshold tuned end-to-end.",
  None,
  "Abstention did not help: the membrane scored 0.613 on content-only prompts against 0.615 for the TRM alone. The errors are bare sentences routed to other adapters, and for those the base model (0.115) is no better than a wrong adapter, so abstaining gains nothing.",
  "Before adding an abstain option, check what the fallback scores on the inputs the router gets wrong.",
  "none")

R(RT, "specialist_routing", "observed", Q1, QP, "3.5 Routing the library, expected utility",
  ["| TRM + expected-utility over the cross-task matrix (cross-fitted) | 0.643 | 0.684 | 0.570 | 0.603 |",
   "off-task, three adapters are near-ties, and the decision chases noise\n   in the estimated matrix"],
  CTX_RT + " The measured adapter-by-environment score matrix is available.", SK_RT,
  "Replace argmax routing with expected-utility routing over the measured cross-task matrix.",
  None,
  "Expected-utility routing was worse: routing accuracy 0.643 and macro 0.684 on full prompts (TRM 0.693), 0.570 and 0.603 on content only (TRM 0.615). Off-task three adapters are near-ties, so the decision chases noise in the estimated matrix.",
  "Cross-fit the matrix so the estimate is not scored on its own data.",
  "none")

R(RT, "specialist_routing", "observed", Q1, QP, "3.5 and 3.4, the costly error",
  ["the costly error (a math problem sent to\n   the reverse-text adapter) is rare (8/200)", "A skill library on one base is therefore only as good as its router.",
   "| gsm8k | 0.845 | **0.045** | **0.875** | 0.850 | 0.875 |"],
  CTX_RT + " The reverse-text adapter starts reversing any input: on GSM8K it scores 0.045 against 0.845 for the base.", SK_RT,
  "Weight router errors by their cost: a misroute into the reverse-text adapter is far costlier than a misroute among the self-distilled adapters.",
  None,
  "The costly error, a math problem sent to the reverse-text adapter, was rare (8 of 200 in the source's count); misroutes among the other adapters cost little because off-task they are close to harmless. A skill library on one base is only as good as its router.",
  "Report the confusion matrix with per-cell cost, not just accuracy.",
  "none")

R(RT, "specialist_routing", "observed", Q1, QP, "Abstract and 3.7, routed library against the 4-bit 8B",
  ["the routed four-skill library scores a macro 0.693 against 0.598 for the 4-bit 8B — in 2.4 GiB of VRAM instead of 5.2 GiB — though on the three tasks the 4-bit model can already do, it stays ahead.",
   "| Q1 Bonsai 8B + all four adapters loaded | **2.37 GiB** | 1.55 GiB |"],
  CTX_RT + " The alternative is a single 4-bit Qwen3-8B (5.23 GiB VRAM) that needs no router.", SK_RT,
  "Serve the four adapters on the 1-bit base behind the TRM router instead of running a single larger 4-bit model.",
  None,
  "With each prompt routed to its adapter the macro score was 0.693 against 0.598 for the 4-bit 8B, in about 2.4 GiB of VRAM instead of 5.2 GiB. The lead comes from reverse-text; on the three tasks the 4-bit model can already do (GSM8K, MedMCQA, IFEval) it stays ahead.",
  "Compare per-environment scores, not only the macro average, because one skill can carry the average.",
  "none")


# ---------------------------------------------------------------------------
# BitAgent ternary-lora-5080: LoRA on frozen ternary LLMs, RTX 5080 Laptop 16 GB
# ---------------------------------------------------------------------------
BA = "ba"
BP = "experiments/ternary-lora-5080/TERNARY_LORA_5080_RESULTS.md"
SK_T = ["llama-finetune-lora on packed TQ2_0 / TQ1_0 ternary weights via QVAC Vulkan backend", "rank-8 LoRA on attention projections, F32 adapter and AdamW state",
        "warmup plus cosine learning-rate schedule", "held-out perplexity with llama-perplexity", "greedy exact match with llama-server on 120 held-out prompts",
        "per-process VRAM monitor"]
CTX_T = ("Fine-tuning LoRA adapters on frozen sub-2-bit ternary models (BitNet XL 1.3B, BitNet 2B4T, Falcon3 3B/7B/10B) on an RTX 5080 Laptop 16 GB under Windows with the QVAC fork of llama.cpp over Vulkan. "
         "Task: a synthetic rule-based instruction set (date formatting, zero-padding, field lookup, check code) with a 120-example held-out set; one seed per run.")

R("ternary-lora-5080/xl-lr-sweep", "quantization_aware_adaptation", "observed", BA, BP, "5. Results",
  ["3.117 → 2.731", "3.117 → 1.698", "2.5 % → **47.5 %**", "Peak process VRAM", "2,082 MiB"],
  CTX_T + " Model: BitNet XL 1.3B in TQ2_0 (whole-file 5.72 bits per weight because the embeddings, output and every ffn_down stay F16).", SK_T,
  "Train a rank-8 adapter with a warmup-plus-cosine schedule, peak lr 2e-4 for 228 steps instead of 1e-4 for 95 steps.",
  None,
  "Held-out perplexity went 3.117 to 2.731 at lr 1e-4 (95 steps, greedy exact match 2.5% to 3.3%) and 3.117 to 1.698 at lr 2e-4 (228 steps, exact match 2.5% to 47.5%), at 2,082 MiB peak process VRAM; the TQ1_0 version reached 3.117 to 1.738 and 49.2%.",
  "Reload the saved adapter in a fresh process and compare perplexity to the in-run value; compute exact match from yet another process.",
  "none")

R("ternary-lora-5080/xl-lr-sweep", "failure_diagnosis", "observed", BA, BP, "5. Results, per-family exact match",
  ["**check code 0 % in every run**", "The check code needs letter-position arithmetic that rank-8 attention-only LoRA at this step count does not learn"],
  CTX_T + " Per-family greedy exact match with the adapter is reported for the best runs.", SK_T,
  "Break exact match down by task family before reading the aggregate as 'the adapter learned the task'.",
  None,
  "Date formatting reached 80-100% and field lookup 97-100%, zero-padding only 17-27%, and the check code 0% in every run. The check code needs letter-position arithmetic that a rank-8 attention-only LoRA at this step count does not learn; the source did not investigate further.",
  "Report per-family exact match next to the aggregate and mark what was not investigated.",
  "none")

R("ternary-lora-5080/divergence", "failure_diagnosis", "observed", BA, BP, "5. Results, Instability found",
  ["diverged after step ~165", "held-out PPL 40.2 vs 3.12 for the base", "Warmup + cosine to 1e-6 at the same peak lr was stable", "The trainer has no gradient clipping option."],
  CTX_T + " First smoke run: constant lr 2e-4, no warmup, 228 steps; loss fell to about 0.8 by step 60.", SK_T,
  "Use warmup plus cosine decay to 1e-6 at the same peak learning rate, since the trainer offers no gradient clipping.",
  None,
  "The constant-lr smoke run learned fast and then diverged after step about 165 (last-10-step mean loss 4.06); the saved adapter had norm 48.9 and held-out perplexity 40.2 against 3.12 for the base. The same peak lr with warmup and cosine to 1e-6 was stable.",
  "Evaluate the saved adapter on the held-out set, not only the training loss curve, and keep the diverged run as evidence.",
  "none")

R("ternary-lora-5080/2b4t-lr", "learning_gain_prediction", "observed", BA, BP, "5. Results, 2B4T",
  ["2B4T needs a higher lr** than XL at the same step budget (2e-4 barely moved it; 6e-4 did)", "3.056 → 2.671", "3.056 → 2.273", "0 % → 0.8 %", "0 % → 26.7 %",
   "With one seed each I can't say whether that is the model or luck."],
  CTX_T + " Model: BitNet b1.58 2B4T converted to TQ2_0, 158 steps, ctx 256.", SK_T,
  "Raise the peak learning rate from 2e-4 to 6e-4 for the 2B4T model at the same 158-step budget.",
  None,
  "At lr 2e-4 perplexity moved 3.056 to 2.671 and exact match 0% to 0.8%; at 6e-4 perplexity moved 3.056 to 2.273 and exact match 0% to 26.7%. With one seed each the source cannot say whether the higher lr requirement is the model or luck.",
  "Rerun with more seeds before generalizing a learning-rate rule across models.",
  "none")

R("ternary-lora-5080/model-scale", "learning_gain_prediction", "observed", BA, BP, "5. Results, scaling to 10.3B",
  ["8.919 → 1.664", "8.333 → 1.664", "8.291 → 1.688", "**4,775 MiB**", "3,629 MiB", "2,087 MiB"],
  CTX_T + " Falcon3 3B, 7B and 10B in TQ2_0, 173 steps at lr 4e-4, ctx 128, batch 1, 16 GB card shared with another session's job.", SK_T,
  "Scale the frozen ternary base from 3B to 7B to 10B with the same adapter recipe at ctx 128.",
  None,
  "Held-out perplexity fell 8.919 to 1.664 (3B, exact match 0% to 56.7%), 8.333 to 1.664 (7B, 6.7% to 55.0%) and 8.291 to 1.688 (10B, 1.7% to 56.7%), at peak process VRAM of 2,087, 3,629 and 4,775 MiB. Throughput fell from 39 to 15 to 11.6 tokens per second.",
  "Record peak process VRAM per model with a per-process monitor (nvidia-smi cannot report it under WDDM) and note that perplexities at ctx 128 are not comparable with ctx 256.",
  "none")

R("ternary-lora-5080/tq1_0-outprod", "failure_diagnosis", "observed", BA, BP, "6. Defects found and fixes, 1",
  ["No Vulkan OUT_PROD kernel for TQ1_0", "12.7 s/step with the GPU at 10–20 % utilisation", "2.8 s/step"],
  CTX_T + " The TQ1_0 (1.69 bits per weight) XL run starts and trains, but slowly with the GPU nearly idle.", SK_T + ["backend op-support inspection", "per-step loss comparison with the TQ2_0 run"],
  "Check which backend ops lack a GPU kernel for the quantization type, then add the missing Vulkan OUT_PROD kernel and compare per-step loss with the TQ2_0 run.",
  None,
  "The scheduler silently fell back to the CPU for the missing op, giving 12.7 s/step with the GPU at 10-20% utilization. Adding out_prod_tq1_0 shaders brought it to 2.8 s/step on the full run. The unpatched CPU-fallback run reproduced the TQ2_0 run's per-step loss to 4-5 digits for 11 steps, as expected for identical ternary values.",
  "Reproduce the sibling quantization's per-step loss on the patched kernel before trusting the speedup.",
  "none")

R("ternary-lora-5080/2b4t-conversion", "failure_diagnosis", "observed", BA, BP, "6. Defects found and fixes, 3",
  ["845 (stock) → 1,488 → 2,152 (partially fixed) → **3.056**", "re-ran weight_quant() on weights already dequantised to ternary×scale",
   "The `bitnet` graph uses SiLU gating; 2B4T uses squared ReLU."],
  "A BitNet b1.58 2B4T checkpoint converted to a ternary GGUF gives a base held-out perplexity of 845 on CPU where the other models give 3 to 9.",
  ["CPU perplexity on the held-out text as a conversion check", "sign agreement with the bf16 master", "per-tensor scale comparison"],
  "Use base perplexity on the held-out text as the check on a conversion, then fix converter defects one at a time and re-measure.",
  None,
  "Perplexity went 845 (stock) to 1,488 and 2,152 (partially fixed) to 3.056 matching the other models. Three defects: weight_quant() re-run on already dequantised weights, a scale divided where 2B4T stores the magnitude (unpacked signs agree with the bf16 master on 99.8% of entries), and the graph using SiLU gating where 2B4T uses squared ReLU.",
  "Compare unpacked signs and layer scales with the bf16 master and re-check base perplexity.",
  "none")

R("ternary-lora-5080/gpu-resets", "failure_diagnosis", "observed", BA, BP, "6. Defects found and fixes, 4 and 5",
  ["the dGPU periodically drops to a ~15 W cap", "exit code 0xC0000409", "`GGML_VK_MAX_NODES_PER_SUBMIT=8` fixes the immediate failures (7B ctx 512) but 10B ctx 512 still crashes every 10–60 steps",
   "Checkpoint-and-resume in the runner gets such a run through."],
  CTX_T + " Falcon3-10B at ctx 512 (9.7 GB) and Falcon3-7B runs crash with Vulkan waitForFences errors and driver resets (nvlddmkm event 153).", SK_T + ["GGML_VK_MAX_NODES_PER_SUBMIT", "checkpoint every 10 steps with --auto-resume"],
  "Separate power-cap resets from long-dispatch timeouts by checking board power at the crash, shorten command buffers with GGML_VK_MAX_NODES_PER_SUBMIT=8, and checkpoint LoRA weights plus AdamW state every 10 steps with automatic resume.",
  None,
  "Under a ~15 W laptop power cap single dispatches exceeded the 2 s WDDM limit and the trainer died with exit code 0xC0000409 (three times, one at step 103 of 173). Failures that track parameters times context at healthy ~110 W were removed for 7B ctx 512 by 8 nodes per submit, but 10B ctx 512 still crashed every 10-60 steps; checkpoint-and-resume got it through after two resets, with the root cause not proven.",
  "Log board power and clocks at each crash and restore optimizer state and learning-rate position when resuming.",
  "none")

R("ternary-lora-5080/script-review", "failure_diagnosis", "observed", BA, BP, "7. The supplied script (reviewed, not run as written)",
  ["The corpus is 8 sentences repeated 96 times, so the trainer's fixed 5 % hold-out contains text identical to the training text.",
   "No learning rate** (default 1e-5) and no schedule", "GPU not pinned."],
  "A supplied PowerShell script fine-tunes a LoRA on a small repeated corpus and reports a validation loss from the trainer's built-in fixed 5% hold-out. The machine exposes two Vulkan devices (AMD iGPU and NVIDIA).",
  ["script review without running it", "external held-out set disjoint by construction"],
  "Review the script's data split and defaults before running it, and replace the built-in hold-out with an external held-out set whose prompt strings are disjoint from training (asserted in code).",
  None,
  "The corpus is 8 sentences repeated 96 times, so the trainer's fixed 5% hold-out contains text identical to the training text and the validation loss is leaked. The script also leaves the learning rate at the 1e-5 default with no schedule, and does not pin the GPU (every real run used GGML_VK_VISIBLE_DEVICES=1).",
  "Assert train and held-out prompt disjointness in the dataset builder and pin the device explicitly.",
  "none")

R("ternary-lora-5080/memory-and-bpw", "quantization_aware_adaptation", "observed", BA, BP, "3. Checkpoints and 5. Memory",
  ["So \"2.06 bits\" or \"1.69 bits\" describes only the linear layers.", "for BitNet XL the ffn_down rows (length 5,460) are not a multiple of the 256-element block, so they cannot be ternary-packed at all",
   "The activation buffer is the part that scales with context and vocabulary, not the ternary weights"],
  CTX_T, SK_T + ["gguf_stats.py actual bits per weight from tensor byte sizes"],
  "Budget training memory from the activation buffer (context times vocabulary), not from the 2-bit weight size, and quote bits per weight from real tensor byte sizes.",
  None,
  "Nominal 2.06 (TQ2_0) or 1.69 (TQ1_0) bits per weight covered only the linear layers; whole-file bits per weight were 5.72 for the XL TQ2_0 file, 3.97 for 2B4T, 3.57 for Falcon3-7B and 3.15 for Falcon3-10B because embeddings and output stay F16, and the XL ffn_down rows (length 5,460) cannot be ternary-packed. For XL at ctx 256 the training compute buffer was 1,068 MiB against 744 MiB of model weights.",
  "Compute bits per weight from the GGUF tensor byte sizes and record the training compute buffer from the load log.",
  "none")

R("ternary-lora-5080/tools-limits", "experiment_design", "observed", BA, BP, "1. What was and was not tested",
  ["Not supported** by `llama-finetune-lora`: one datapoint (`-c` tokens) is one micro-batch is one optimizer step.", "adapter weights, gradients and AdamW state are F32",
   "I did **not** repeat any run, so run-to-run determinism and seed variance are unmeasured."],
  CTX_T + " The request asked for gradient accumulation, gradient checkpointing, fp16 adapter computation, a fixed seed, and 50-100 step first runs.", SK_T,
  "Check the trainer's actual feature set first and report unsupported requests as such: no gradient accumulation, no gradient checkpointing, F32 adapter state, a fixed 5% validation split, and no repeated runs.",
  None,
  "The trainer supports none of gradient accumulation, gradient checkpointing or half-precision adapters; --lora-seed fixes adapter init but no run was repeated, so seed variance is unmeasured, and every result is single-seed on synthetic rule-based data that says nothing about downstream quality.",
  "State in the report which requested controls were unavailable and which results are single-seed.",
  "none")

R("ternary-lora-5080/out-prod-tiled", "failure_diagnosis", "hypothesis", BA, BP, "11. In progress: removing the long-context crash",
  ["Vulkan's OUT_PROD (the backward matmul through the frozen quantised weights) uses a one-thread-per-output-element shader that loops over the whole reduction dimension",
   "It has not been tested"],
  CTX_T + " Long-context crashes track parameters times context, not VRAM, at healthy power, and disappear with smaller command buffers; the checkpoint-and-resume workaround does not remove the cause.", SK_T,
  "Force the tiled OUT_PROD shader on NVIDIA with GGML_VK_OUT_PROD_TILED=1 (upstream already uses it on AMD RADV to avoid VK_ERROR_DEVICE_LOST from slow threads) and rerun the 10B ctx 512 job.",
  "Hypothesis from reading the backend, untested when written: the one-thread-per-element shader makes single dispatches long enough to cross the 2 s limit, so the tiled shader should remove the crashes.",
  "Run the 10B ctx 512 job with and without the flag and count device-lost events per 100 steps.",
  "proposed")

# ---------------------------------------------------------------------------
# HybridTRMLDT S0: Bonsai 2 ternary 27B (PTQ1_0) trainability gate
# ---------------------------------------------------------------------------
S0 = "hy_s0"
S0D = "research/sovereign_learning/s0_trainability/"
S0L = "hybrid-trm-ldt/s0-bonsai2"
CTX_S0 = ("Target: Ternary-Bonsai-2-27B in PrismML's PTQ1_0 GGUF type (128 ternary values per 28-byte group, FP16 scale, input-axis Hadamard-1024 rotation with explicit sign vectors). "
          "S0 is a kill test: can a frozen low-bit base carry an unmerged LoRA adapter, with train and deploy identity, before any learned routing is built? Reference code is a CPU codec and a PyTorch oracle pinned to a PrismML llama.cpp fork.")
SK_S0 = ["standard-library PTQ1_0/PQ2_0 codec", "mmap GGUF reader", "fork's compiled dequantize_row_ptq1_0 through ctypes", "FrozenTernaryLinear PyTorch oracle with unmerged LoRA",
         "llama-perplexity KL-divergence mode", "llama-bench with nvidia-smi VRAM sampling", "vramorama lease on the shared GPU"]

R(S0L, "quantization_aware_adaptation", "observed", S0, S0D + "README.md", "Real-row parity, measured 2026-10-31",
  ["Layout: 851 tensors (402 PTQ1_0, 353 F32, 96 BF16), GGUF v3, alignment 32.", "Rows: 0 mismatches, bit-exact.", "About 67% of the weights are non-zero.",
   "Still open in step 1: the runtime order of sign and transform application, and tokenization."],
  CTX_S0 + " Checkpoint file Ternary-Bonsai-2-27B-PTQ1_0.gguf (5,946,648,928 bytes, sha256 pinned to the Hugging Face LFS hash).", SK_S0,
  "Decode real PTQ1_0 rows with the Python reference codec and compare them with the fork's compiled C decoder, sampling every PTQ1_0 tensor and comparing two tensors in full.",
  None,
  "Zero mismatches, bit-exact: every PTQ1_0 tensor was sampled (first, last and 2 seeded random rows) and blk.0.attn_gate and blk.63.ffn_down were compared in full; layout was 851 tensors (402 PTQ1_0, 353 F32, 96 BF16) with non-overlapping payloads ending exactly at end of file, and about 67% of weights non-zero. The runtime order of sign application and the transform, and tokenization, remained open.",
  "Compare against the runtime's own decoder on the real file, so a layout misreading shared by two implementations would show up, and list what the check does not cover.",
  "none")

R(S0L, "quantization_aware_adaptation", "observed", S0, S0D + "README.md", "Projection parity, measured 2026-11-01",
  ["its pass criteria were committed (`ca059ec`) before the first run", "4.1e-7 | 2.0e-6 | 1.000 | 0.0054 / 1.03 | PASS",
   "the fork's own Q8_0 activation quantization (ratio C = 1.000): about 0.5% of output rms per projection"],
  CTX_S0 + " The fork's runtime builds the graph for a Hadamard-folded weight: multiply by signs, block-1024 FWHT as a mul_mat, then mul_mat with the PTQ1_0 weight.", SK_S0,
  "Commit pass thresholds first (FWHT versus oracle below 1e-5, fork F32 versus oracle below 1e-4, ratio C at most 1.05), then compare the fork's projection with the PyTorch oracle on seeded Gaussian inputs for three tensors and two CPU kernel sets.",
  "Registered thresholds: A < 1e-5, B < 1e-4, C <= 1.05.",
  "All three tensors passed: for blk.0.attn_gate A was 4.1e-7 and B 2.0e-6 with C = 1.000. The only gap between the fork's path and the oracle is the fork's Q8_0 activation quantization, about 0.5% of output rms per projection (cosine 0.99999), which compounds over 64 layers and can be judged against the provisional 0.03 full-logit tolerance only after a whole-model run.",
  "Use real signs, state which build and kernels were tested, and list what is not covered (activations, CUDA backend, ssm_out weights, full-model logits).",
  "preregistered")

R(S0L, "learning_gain_prediction", "observed", S0, S0D + "logit_parity/AMENDMENT_A1.md", "A1 results (2026-11-02)",
  ["| PQ2_0 CPU vs PTQ1_0 CPU | 99.326% ± 0.091 | 0.000272 | 0.837389 | 0.005110 | 0.739% | 0.999930 |",
   "Unlike the mixed run, PQ2_0 and PTQ1_0 are not bit-identical on the CPU path", "This is unconfirmed, because the logs do not record per-token data.",
   (S0D + "logit_parity/PREREG.md", "PQ2_0 on CPU should be very close to bit-identical")],
  CTX_S0 + " Full-model logit parity on 32 chunks of wikitext-2 (ctx 512) with llama-perplexity; the reference is PTQ1_0 on CPU. Pass 2 compares the PQ2_0 packing of the same trits and scales, whose dequantized weights are bit-identical.", SK_S0,
  "Run the pre-registered comparison PQ2_0 on CPU against PTQ1_0 on CPU, primary criterion top-1 agreement at least 99.0%.",
  "Expectation recorded in the pre-registration so a miss would be visible: PQ2_0 on CPU should be very close to bit-identical, since both formats feed the same trits and scales into Q8_0-activation dot products.",
  "Top-1 agreement was 99.326% ± 0.091 (55 of 8,160 tokens disagree), mean KLD 0.000272 but maximum KLD 0.837, perplexity ratio 0.999930. The primary criterion passed but the bit-identical expectation was wrong: the CPU kernels of the two formats differ numerically even though the dequantized weights are identical. The source suspects PTQ1_0's CPU kernel but marks it unconfirmed because per-token data was not logged.",
  "Report top-1 agreement with its standard error, mean and 99.9th-percentile and maximum KLD, and keep expectation misses visible rather than rewriting them.",
  "preregistered")

R(S0L, "failure_diagnosis", "observed", S0, S0D + "logit_parity/AMENDMENT_A1.md", "What happened; Change",
  ["Passes 1 and 2 used `-ngl 0`, but this PrismML build defaults to `--op-offload` on.", "The reference is a mixed CPU+CUDA placement, not the CPU path",
   "Passes 1 and 2 are rerun with `-ngl 0 --device none --no-op-offload`"],
  CTX_S0 + " The pre-registered logit-parity run finished with every pass exiting 0 and PQ2_0 versus PTQ1_0 showing 100.000% top-1 agreement; the reference was meant to be CPU-only.", SK_S0 + ["amendment written after the first run and before any rerun"],
  "Read the run log for device assignment before calling a reference CPU-only, then rerun passes 1 and 2 with -ngl 0 --device none --no-op-offload in a fresh work directory under a dated amendment that keeps the as-run results.",
  None,
  "Passes 1 and 2 used -ngl 0, but the build defaults to --op-offload on, and the log showed fused Gated Delta Net ops assigned to CUDA0, so the reference was a mixed CPU+CUDA placement. As-run results were kept (PQ2_0 and PTQ1_0 agreed 100.000% under identical placement); the rerun then showed 99.326% on the true CPU path, so the first result had hidden a real CPU kernel difference.",
  "Make the runner report 'CPU-only' only if the log shows no CUDA device assignment, and keep both the as-run and rerun tables.",
  "none")

R(S0L, "quantization_aware_adaptation", "observed", S0, S0D + "logit_parity/AMENDMENT_A1.md", "A1 results, CUDA versus CPU",
  ["| PTQ1_0 full CUDA (`-ngl 99`) vs PTQ1_0 CPU | 99.326% ± 0.091 | 0.000196 | 0.120659 | 0.005064 | 0.465% | 1.000105 |",
   "This establishes CPU-versus-CUDA logit parity for PTQ1_0 on this build.", "CPU-only reference perplexity: 7.4935 ± 0.2119."],
  CTX_S0 + " The training backend (CUDA) and a CPU reference must produce the same next-token distribution before adapter results can be trusted across them.", SK_S0,
  "Compare PTQ1_0 on full CUDA (-ngl 99, under a 9 GB vramorama lease) with the CPU-only PTQ1_0 reference on the same 32 chunks.",
  None,
  "Top-1 agreement 99.326% ± 0.091, mean KLD 0.000196, max KLD 0.1207, 99.9th percentile 0.005064, RMS change in probability 0.465%, perplexity ratio 1.000105 (CPU-only reference perplexity 7.4935 ± 0.2119). The primary criterion passed, establishing CPU-versus-CUDA logit parity on this build.",
  "Run the CUDA pass inside a GPU lease and compare against a reference whose placement the log confirms.",
  "none")

R(S0L, "quantization_aware_adaptation", "observed", S0, S0D + "gpu_baseline/README.md", "Result (2026-11-02)",
  ["| 32,768 | 950 ± 54 | 47.7 ± 0.9 | 8,480 |", "Peak VRAM at 32K is 8.3 GiB, which is under the S0 go criterion of 10 GB.",
   "8K was the first test, and its repetitions include the GPU clock ramp."],
  CTX_S0 + " llama-bench with full offload (-ngl 99 -fa on -p 512 -n 128 -r 3) at context depths 8,192 and 32,768 on an RTX 5080 Laptop 16 GB with nvidia-smi sampling every 500 ms, inside a vramorama lease; S0 go needs at most 10 GB inference VRAM at 32K.", SK_S0,
  "Measure inference throughput and peak VRAM at two context depths with sampling, and use the second depth's generation figure because the first includes the GPU clock ramp.",
  None,
  "At 8,192 context: prompt 1,234 ± 33 tok/s, generation 40.5 ± 24.4, peak VRAM 6,976 MiB. At 32,768: prompt 950 ± 54, generation 47.7 ± 0.9, peak 8,480 MiB (8.3 GiB), under the 10 GB go criterion. The 8K generation spread is a clock ramp; the 32K figure is the stable one.",
  "Check for other GPU processes before the run and discard invalid sensor readings (one power sample read 628,233 W).",
  "none")

R(S0L, "experiment_design", "hypothesis", S0, S0D + "logit_parity/PREREG.md", "Pre-registration (2026-11-01)",
  ["Written and committed before any logit pass ran.", "A failed pass is recorded as it ran, with its log. Nothing is rerun with changed settings without an amendment.",
   "It is **not** oracle parity: the branch has no independent full-model forward.", "Expectation, recorded so a miss is visible"],
  CTX_S0 + " Question: does the runtime give the same next-token distribution when the backend changes from CPU to CUDA and when the packing changes from PTQ1_0 to PQ2_0?", SK_S0,
  "Pre-register the passes, fixed inputs (pinned runtime, model hashes, text hash, ctx 512, 32 chunks), the primary criterion (top-1 agreement at least 99.0%) and an explicit expectation, and say what the test is not (not oracle parity).",
  "Expectation recorded: PQ2_0 on CPU should be very close to bit-identical to PTQ1_0 on CPU; CUDA should differ by kernel and accumulation order only.",
  "Commit the document before any pass runs; record failed passes as they ran; allow changes only through a dated amendment.",
  "preregistered")

R(S0L, "quantization_aware_adaptation", "hypothesis", S0, S0D + "README.md", "Measured S0 acceptance gate, still open",
  ["S0 go requires 99% top-1 agreement, measured max-logit discrepancy below the preregistered bound, <=14 GB peak training VRAM, >=300 train tokens/s and <=10 GB inference VRAM at 32K.",
   "All criteria remain **NOT_EVALUATED**.", "single** 2048-token frozen-base adapter backward pass with a true packed-resident kernel",
   "If any step fails, stop and record it."],
  CTX_S0 + " Steps 1 to 4 of the gate: pin hashes and decode real rows; compare one projection then full-model logits with and without an unmerged rank-16 GGUF adapter; measure 8K/32K inference VRAM and one 2048-token backward pass; report VRAM, tokens/s and hashes.", SK_S0,
  "Run the gate in order on the RTX 5080: logit parity with and without an unmerged rank-16 adapter, 8K/32K inference VRAM, then one 2048-token frozen-base adapter backward pass with a true packed-resident kernel; stop and record at the first failure, with no loops or paid teacher calls.",
  "Go criteria: 99% top-1 agreement, max-logit discrepancy below the preregistered bound, at most 14 GB peak training VRAM, at least 300 training tokens/s, at most 10 GB inference VRAM at 32K; the source marks all as not yet evaluated.",
  "An autograd oracle in the directory does not establish a packed-resident backward kernel; measure it on the card.",
  "preregistered")

# ---------------------------------------------------------------------------
# HybridTRMLDT master: typed membrane benchmarks, sequencer control, gaming vs improvement
# ---------------------------------------------------------------------------
HM = "hy_master"
HP = "papers/hybrid_ldt_trm_draft.md"
FP = "packages/fable_hybrid_architecture_space_20260716/V2_ABOVE_MAJORITY_RESULT.md"
SK_H = ["TRM-style latent proposer", "LDT-style explicit lattice state", "typed membrane (certify, reject or soft-store by provenance)",
        "hard lattice filter", "confidence arbitration", "external fallback controller"]

R("hybrid-trm-ldt/gaming-vs-improvement", "failure_diagnosis", "observed", HM, FP, "Result",
  ["Every one of the 30 adaptive evidence by rejection by seed trajectories\ndegrades oracle proposal behavior.", "The exposed probe produces strict evasion in two seeds and mixed degradation\nin one:",
   "| 211 | +0.508 | -0.0006 | +0.0156 | -0.1089 | -0.1328 |", "| 227 | +0.578 | -0.0149 | -0.0312 | -0.0516 | -0.1875 |"],
  "A proposer is adapted for five rounds against evidence gates (an exposed probe, hidden audits, exact mechanics) with accepted-only self-imitation. Registered readouts are gate passage, proposal utility, exact soundness, hidden-audit AUROC and the oracle-optimal rate, over seeds 211, 223 and 227.",
  SK_H + ["accepted-only self-imitation learner", "hidden audit probe"],
  "Read passage rate against hidden-audit AUROC and oracle-optimal rate together before calling a rise in gate passage an improvement.",
  None,
  "No tested gate improved the proposer: all 30 adaptive trajectories degraded oracle proposal behaviour. Gate passage rose by +0.508, +0.289 and +0.578 while proposal utility fell (-0.0006, -0.0166, -0.0149), hidden-audit AUROC fell and the oracle-optimal rate fell by 0.125 to 0.19. The exposed probe produced strict evasion in two seeds and mixed degradation in one.",
  "Always keep a hidden audit the proposer cannot see and compare gate passage with it.",
  "none")

R("hybrid-trm-ldt/gaming-vs-improvement", "learning_gain_prediction", "observed", HM, FP, "Power Caveat Closed",
  ["The three final proposers score 0.789 / 0.789 / 0.797 at round zero against majority baselines\n0.680 / 0.672 / 0.734", "clear the registered +0.020 margin",
   "The experiment can therefore observe proposal improvement"],
  "Version 1 of the gaming-versus-improvement experiment had a constant-majority proposer, so a flat or degraded result might only reflect no room to improve. Version 2 uses proposers above the majority baseline.",
  SK_H,
  "Before testing whether gates improve the proposer, require each proposer to beat its majority baseline by the registered +0.020 margin, keep at least 0.203 to the oracle ceiling, and show action-movement capacity under five adaptation rounds.",
  "Registered: the proposer clears the majority baseline by +0.020 and can still move.",
  "Round-zero scores were 0.789, 0.789 and 0.797 against majority baselines 0.680, 0.672 and 0.734, so all three cleared the margin; the power caveat is closed and v1's flat result cannot be attributed to a constant-majority limitation. The subsequent adaptation still degraded every trajectory.",
  "State the baseline margin and the headroom to the ceiling before the adaptation runs.",
  "preregistered")

R("hybrid-trm-ldt/gaming-vs-improvement", "skill_composition", "observed", HM, FP, "Result and Architectural Update",
  ["Authorization: exact mechanics can safely determine whether a proposal may execute.", "Correction: a distinct fallback can recover utility when authorization rejects a proposal.",
   "Teaching: neither authorization nor correction trains the proposer unless the corrective action is infused\n   as an independently verified target.", "The exposed gate's controller benefit falls from +0.0042 to +0.0006"],
  "Exact mechanics can reject unsafe proposals and a distinct fallback can supply a corrected action, but the registered learner imitates only accepted proposals and never learns the fallback action on rejected states.",
  SK_H,
  "Assign three separate roles: exact mechanics authorize, a distinct fallback corrects, and a correction-infused learner is taught only from independently verified corrective targets, with strict causal separation between verifier, fallback and learner.",
  None,
  "Executed utility rose by +0.0242 with exact mechanics plus a distinct fallback, but only by correcting a proposer whose own utility had fallen; the exposed gate's controller benefit fell from +0.0042 to +0.0006 (about 85%) as action change fell from 0.549 to 0.104. Authorization and correction do not teach.",
  "Report proposer utility and executed utility separately.",
  "none")

R("hybrid-trm-ldt/typed-membrane-benchmarks", "skill_composition", "observed", HM, HP, "7. Benchmark Summary",
  ["| Sudoku | 0.333 | 1.000 | 1.000 | TRM, Hybrid |", "| Storyworld | 1.000 | 0.172 | 1.000 | LDT, Hybrid |", "TRM helps when proposal/search/routing is needed.",
   "LDT helps when mechanics are checkable.", "Hybrid helps when proposal can be followed by checkable pruning or safety checks."],
  "Five lightweight benchmarks (Sudoku, ARC-1, ARC-2, environment-pointer routing, a coupled storyworld playing task) compare a TRM-style proposer, an LDT-style explicit lattice and a hybrid with a typed membrane.",
  SK_H,
  "Pair a learned proposer with a checkable lattice: the proposer suggests, the lattice certifies or rejects only where mechanics are checkable.",
  None,
  "Sudoku: LDT 0.333, TRM 1.000, hybrid 1.000. Storyworld: LDT 1.000, TRM 0.172, hybrid 1.000. ARC-1 and ARC-2 tied at 1.000. Routing: TRM 0.815, hybrid 0.815 (see routing record). TRM helps when proposal, search or routing is needed, LDT when mechanics are checkable, and the hybrid when a proposal can be followed by checkable pruning or safety checks; it is not automatically better.",
  "Report all three arms on every benchmark, including the ones where the hybrid only ties.",
  "none")

R("hybrid-trm-ldt/storyworld-playing", "failure_diagnosis", "observed", HM, HP, "7. Benchmark Summary, storyworld confidence/type split",
  ["| Secret ending | Confidence arbitration | 0.938 | 0.94 |", "| Secret ending | Typed membrane | 1.000 | 1.00 |",
   "Confidence arbitration sees high local margins and repeatedly selects `defuse`", "fails because evidence never reaches the secret gate"],
  "In a storyworld playing task a secret ending requires evidence to reach a threshold while heat is managed. Two control policies are compared: confidence arbitration (follow the highest local margin) and a typed membrane that preserves reachability of the secret-ending predicate.",
  SK_H,
  "Read the failed transcript step by step and compare the state trajectory with the predicate the run needed, before adjusting the policy.",
  None,
  "Success was 1.000 for the typed membrane and 0.938 for confidence arbitration on the secret ending. In one failing run confidence arbitration saw high local margins and chose defuse four times (ending at trust 3, evidence 1, heat 0, scene 5) so evidence never reached the secret gate; the typed membrane followed defuse, wait, wait, investigate, defuse, investigate and satisfied the predicate.",
  "Check reachability of the goal predicate along the trajectory, not only local margins.",
  "none")

R("hybrid-trm-ldt/sequencer-control", "skill_composition", "observed", HM, HP, "Abstract",
  ["improves family-macro utility by 0.0069 (95% context-clustered bootstrap interval [0.0049,0.0092], Holm-adjusted p=0.0006)",
   "0.0003 advantage over lineage-only control is not significant (p=0.5023)", "This measures sequencer control over deterministic skills, not neural weight infusion"],
  "Fixed hybrid skill sequencers over deterministic skills are replayed on 667 registered episodes. Controls: global signed sequencing, lineage-only control and topology-controlled localization.",
  ["global signed sequencing", "lineage-only control", "topology-controlled localization", "context-clustered bootstrap with Holm adjustment"],
  "Localize sequencing control by topology rather than applying one global signed sequence, and compare with a lineage-only control.",
  None,
  "Topology-controlled localization improved family-macro utility over global signed sequencing by 0.0069 (95% context-clustered bootstrap interval [0.0049, 0.0092], Holm-adjusted p = 0.0006) with unchanged accuracy and lower cost. Its 0.0003 advantage over lineage-only control was not significant (p = 0.5023). The result measures sequencer control over deterministic skills, not neural weight infusion or leaderboard performance.",
  "Cluster the bootstrap by context and adjust for multiple comparisons; compare against the strongest simpler control.",
  "none")

R("hybrid-trm-ldt/forecast-integrity", "skill_composition", "observed", HM, HP, "Abstract",
  ["digest sealing preserves\n100% clean acceptance while reducing negative-control acceptance from 44.4% under topology-only arbitration to\nzero",
   "This is a deterministic authorization result, not evidence for learned AIRIS robustness.", "Storyworld precision\nfalls to 65.6%--78.1%"],
  "Calibration-derived sequence rules are passed through an AIRIS/DAS-shaped retrieval service. Across 6,670 paired clean and fault-injection trials, the question is whether forecast transport integrity can be separated from model quality.",
  ["digest sealing of forecasts", "topology-only arbitration", "fault-injection negative controls", "independently induced context rules"],
  "Seal forecasts with a digest so tampered or unauthorized forecasts are rejected deterministically, and test with paired clean and fault-injection trials.",
  None,
  "Digest sealing kept 100% clean acceptance while cutting negative-control acceptance from 44.4% under topology-only arbitration to zero. Independently induced context rules reached 94.0% held-out episode-winner precision but reproduced the frozen controller exactly; storyworld precision fell to 65.6-78.1%, motivating state-conditioned typed fallback. This is a deterministic authorization result, not evidence of learned robustness.",
  "Report acceptance on clean and negative-control trials separately.",
  "none")

R("hybrid-trm-ldt/env-pointer-routing", "specialist_routing", "observed", HM, HP, "6.3 Environment-Pointer Routing",
  ["Hybrid hard filter: 0.584", "forcing token-lattice candidates as hard filters reduces accuracy from 0.815 to 0.584", "This evidence is not environment-sound. It is model/experience-derived."],
  "Routing maps a prompt to an environment ID. A TF-IDF-plus-MLP style TRM router is compared with an LDT that builds token-derived candidate sets C_t as the intersection of environment sets E_w over the prompt's tokens, on a 173-prompt benchmark.",
  ["lexical TRM analogue", "token-lattice candidate sets", "soft guidance (argmax over C_t if the TRM top route is in C_t, else TRM argmax)", "hard filter"],
  "Use the learned lattice only as soft guidance: pick the TRM's best route inside the candidate set when the TRM's top route is in it, otherwise keep the TRM's top route.",
  None,
  "LDT alone 0.474, TRM 0.815, soft hybrid 0.815, hard-filter hybrid 0.584. Forcing token-lattice candidates as hard filters cut accuracy from 0.815 to 0.584 because the lattice evidence is learned from data and not environment-sound. The hybrid only ties the TRM here.",
  "Evaluate the hard-filter variant explicitly; do not eliminate routes on lexical evidence that is not sound.",
  "none")

R("hybrid-trm-ldt/env-pointer-routing", "specialist_routing", "observed", HM, HP, "7. Benchmark Summary, routing architecture variants",
  ["| Typed membrane | 0.815 | 141/173 | soft telemetry unless sound |", "| Hard gate | 0.584 | 101/173 | LDT candidates hard-filter TRM |",
   "| Confidence arbitration | 0.815 | 141/173 | train-selected `gamma=0.0` |", "| Noisy lexical evidence | Typed membrane | Keep learned candidates soft |"],
  "Three control policies sit between a TRM router and lattice evidence: a typed membrane, a hard gate and confidence arbitration with a train-selected gamma.",
  SK_H,
  "Choose the control policy by the regime: typed membrane for noisy lexical evidence and checkable search spaces, hard gate only when the lattice state is environment-sound, confidence arbitration for soft preference objectives.",
  None,
  "Accuracy: typed membrane 0.815 (141/173, soft telemetry unless sound), hard gate 0.584 (101/173), confidence arbitration 0.815 (141/173, train-selected gamma = 0). The source's rule of thumb: with noisy lexical evidence keep learned candidates soft.",
  "Report the correct count (out of 173) per policy so ties are visible.",
  "none")

# ---------------------------------------------------------------------------
# TRMStoryworld
# ---------------------------------------------------------------------------
TS = "trmsw"
R("trmstoryworld/swmd-min-bench", "task_graph_navigation", "observed", TS, "mechinterp/CODEX_LOG.md", "Turn 8 (2026-02-11)",
  ["Verified aggregate manifest stats: worlds=40, ok=40, steps avg=35.75, terminal_nodes_total=6, special_hits_total=0.",
   "Ran a strict sequential TRM benchmark across all 40 SWMD-min worlds with 3 episodes and 200-step cap"],
  "Forty minified storyworld graphs (SWMD-min) are rolled out by a deterministic TRM controller with latent tracing, 3 episodes per world and a 200-step cap, to benchmark navigation through each world's graph toward terminal and special nodes.",
  ["TRM trace benchmark with latent trace output", "SWMD-min parser and deterministic rollout path", "SAEv1 artifacts"],
  "Run the controller sequentially on all 40 worlds and aggregate steps, terminal-node and special-node counts, and define each counter before using it as evidence about navigation.",
  None,
  "All 40 worlds ran (ok=40) with an average of 35.75 steps, far below the 200-step cap. The log reports terminal_nodes_total=6 and special_hits_total=0 across the batch without defining terminal_nodes_total beyond its name, so the aggregate is a baseline to compare against, not evidence about reachability of endings.",
  "Define each aggregate counter in the manifest and report it next to the number of worlds that merely completed.",
  "none")

R("trmstoryworld/trm-sw1", "failure_diagnosis", "observed", TS, "negotiation-storyworld/logs/trm_sw1_summary.json", "whole file",
  ["\"episodes\": 5", "\"outcomes\": {\"maneuver\": 5}", "\"coalition_recommendation_steps\": 30", "\"defection_recommendation_steps\": 30"],
  "A negotiation-storyworld controller run logs five episodes of six steps each, with a summary of outcomes and of recommendation counts.",
  ["episode summary JSON", "per-step recommendation trace"],
  "Open the per-step recommendations before treating the outcome counts as evidence about the policy.",
  None,
  "The summary shows 5 episodes and 30 steps, every episode ending in the same outcome (maneuver: 5), and both a coalition and a defection recommendation on all 30 steps. A summary in which both recommendations fire at every step cannot show which one drove the outcome.",
  "Log which recommendation was taken at each step and compare against a policy that always takes one of them.",
  "none")

R("trmstoryworld/determinism-contract", "experiment_design", "hypothesis", TS, "AGENTS.md", "Determinism Rules",
  ["Normalize newlines to LF before hashing.", "Stable sort by IDs (encounter/option/reaction).",
   "Receipt hash payloads must be versioned (`SWMDR|...`, `SFT1|...`) and immutable once published."],
  "A controller dataset is built from SWMD storyworld IR files and evaluated by rollouts whose traces and outputs are committed with receipt hashes.",
  ["canonical JSON with sorted keys and compact separators", "versioned receipt hash payloads", "rows Merkle root and run identity hash"],
  "Make the dataset and rollout receipts deterministic: LF-normalize before hashing, stable-sort by IDs, serialize canonical JSON, and version every receipt payload so published ones stay immutable.",
  "Not a numeric prediction: a procedure that lets two builds of the same input produce identical hashes.",
  "Rebuild the dataset from the same SWMD hash and compare the dataset hash, rows Merkle root and run identity hash.",
  "proposed")


# ---------------------------------------------------------------------------
# EvalMath: eval-reuse toy runs with sealed pre-registered predictions
# ---------------------------------------------------------------------------
EM = "em"
EL = "evalmath/eval-reuse-toys"
EP1, EP2 = "output/eval_reuse_toys/PREREG.md", "output/eval_reuse_toys/PREREG_2.md"
ER1, ER2 = "output/eval_reuse_toys/results.json", "output/eval_reuse_toys/results2.json"
SK_EM = ["seeded JavaScript simulation (plain Node)", "sealed hashes of the prediction file and the script before the run",
         "FRESH, REUSE-NAIVE and REUSE-GATED evaluation arms", "gap = reported score on the reused set minus population accuracy"]
CTX_E1 = ("Toy model of a self-improvement loop that selects candidates on a reused held-out set with score-only feedback. Items have difficulty b ~ N(0, 1.5^2) and a quirk u ~ N(0, 1); "
          "a model solves item i iff a - b_i + u_i > 0; population accuracy is Phi(a / sqrt(1.5^2 + 1)) and starts at 0.26. 100 generations of K = 16 candidates; arms FRESH (new items each generation), "
          "REUSE-NAIVE (one fixed set of n items) and REUSE-GATED (accept only gains above 3 sqrt(p(1-p)/n)); n in {500, 1000, 4000, 16000}; 20 seeds per cell. Deterministic, no language model.")
CTX_E2 = ("Second toy, item-level feedback with a real learner: x ~ N(0, I_6), clean label sign(x0 + 0.7 x1 x2 - 0.3), observed label flipped with probability 0.10 (noise ceiling 0.90), "
          "3-nearest-neighbour learner starting with 100 points and adding 10 per generation for 100 generations. Arms CLEAN (fresh draws), SCORE (best of 16 fresh batches on the reused set), "
          "LEAK (add the 10 items the incumbent currently fails on the reused set, with their observed labels) and MIXED (5 failed plus 5 fresh); reused-set size n in {250, 1000, 4000, 16000}; "
          "sealed test set of 5000; 10 paired seeds per cell.")
RUN1 = [
    ("P1", ["P1. REUSE-NAIVE gap at generation 100, n = 4000, lies in [0.08, 0.25]."], '"P1":{"value":0.035,"range":[0.08,0.25],"pass":false}',
     "REUSE-NAIVE gap at generation 100 with n = 4000 lies in [0.08, 0.25].",
     "The measured gap was 0.035, below the registered range: the prediction failed (the run overpredicted the gap)."),
    ("P2", ["P2. REUSE-NAIVE gap at generation 100 scales as n^(-1/2): log-log slope", "across the four n values lies in [-0.6, -0.4]."], '"P2":{"slope":-0.507',
     "The REUSE-NAIVE gap at generation 100 scales as n^(-1/2): the log-log slope across the four n values lies in [-0.6, -0.4].",
     "The slope was -0.507, inside the registered range: passed."),
    ("P3", ["Phantom share of reported gain over generations 51-100,", "exceeds 0.5 in REUSE-NAIVE for"], '"P3":{"values":[1,0.914,0.771]',
     "The phantom share of reported gain over generations 51-100 (1 - true gain / reported gain) exceeds 0.5 in REUSE-NAIVE for n in {500, 1000, 4000}.",
     "The phantom shares were 1, 0.914 and 0.771 for n = 500, 1000 and 4000, all above 0.5: passed."),
    ("P4a", ["P4a. At n = 4000, REUSE-GATED gap at generation 100 is at most half the", "REUSE-NAIVE gap."], '"P4a":{"gated":0.004,"naive":0.035',
     "At n = 4000 the REUSE-GATED gap at generation 100 is at most half the REUSE-NAIVE gap.",
     "The gated gap was 0.004 against 0.035 for the naive arm, well under half: passed."),
    ("P4b", ["P4b. At n = 4000, final population accuracy orders", "FRESH > REUSE-NAIVE > REUSE-GATED."], '"P4b":{"fresh":0.635,"naive":0.618,"gated":0.309',
     "At n = 4000 final population accuracy orders FRESH > REUSE-NAIVE > REUSE-GATED.",
     "Final population accuracy was 0.635 (FRESH), 0.618 (REUSE-NAIVE) and 0.309 (REUSE-GATED): the ordering held. REUSE-GATED barely moved from the 0.26 start (0.309)."),
    ("P5", ["P5. At n = 4000 in REUSE-NAIVE, the late/early gain ratio", "least twice the same ratio computed from population accuracy."], '"P5":{"reported":0.032,"true":0.008',
     "At n = 4000 in REUSE-NAIVE, the late/early gain ratio computed from reported scores is at least twice the same ratio computed from population accuracy.",
     "The check recorded 0.032 for the reported score and 0.008 for the true accuracy and was marked passed."),
]
for _k, _pq, _rq, _pred, _out in RUN1:
    R(EL, "learning_gain_prediction", "observed", EM, EP1, "Run 1 prediction " + _k,
      _pq + [(ER1, _rq)],
      CTX_E1, SK_EM,
      "Run the sealed score-only reuse simulation and score registered prediction %s against its range." % _k,
      "Registered before the first run (hashes of this file and sim.js sealed first): " + _pred,
      _out,
      "Compare the computed %s statistic with the registered range or ordering; do not adjust the ranges after seeing the output." % _k,
      "preregistered")

RUN2 = [
    ("P1", ["P1. LEAK gap at generation 100: n = 16000 in [0.025, 0.065];", "n = 4000 in [0.10, 0.24]; for n = 1000 and n = 250 reported >= 0.97"],
     '"P1":{"gap16000":0.015,"gap4000":0.076,"rep1000":0.999,"rep250":0.999,"pass":false}',
     "The LEAK gap at generation 100 is in [0.025, 0.065] for n = 16000 and in [0.10, 0.24] for n = 4000, and for n = 1000 and n = 250 the reported score is at least 0.97, above the 0.90 noise ceiling.",
     "The gaps were 0.015 (n = 16000) and 0.076 (n = 4000), both below their ranges, while the reported score at n = 1000 and n = 250 was 0.999: the prediction failed because the gap magnitudes were overpredicted, though the above-ceiling reported scores were right."),
    ("P2", ["P2. LEAK gap dilutes linearly in n while unexhausted: slope", "ln(gap_16000 / gap_4000) / ln 4 in [-1.15, -0.75]."], '"P2":{"slope":-1.182,"range":[-1.15,-0.75],"pass":false}',
     "The LEAK gap dilutes linearly in n while unexhausted: ln(gap_16000 / gap_4000) / ln 4 lies in [-1.15, -0.75].",
     "The slope was -1.182, just steeper than the registered range: failed."),
    ("P3", ["P3. SCORE replicates run 1: gap at n = 4000 in [0.03, 0.15]; log-log slope", "over the four n in [-0.65, -0.35]."], '"P3":{"gap4000":0.041,"slope":-0.516,"pass":true}',
     "SCORE replicates run 1: the gap at n = 4000 is in [0.03, 0.15] and the log-log slope over the four n is in [-0.65, -0.35].",
     "The gap at n = 4000 was 0.041 and the slope -0.516, both inside their ranges: passed. Score-only reuse reproduced with a real learner."),
    ("P4", ["P4. Patches generalize worse than fresh data. At n = 16000, true accuracy", "at generation 100 orders CLEAN > MIXED > LEAK, and LEAK's true gain is less", "than half of CLEAN's."],
     '"P4":{"clean":0.776,"mixed":0.732,"leak":0.696,"cleanGain":0.062,"leakGain":-0.017,"pass":true}',
     "Patches generalize worse than fresh data: at n = 16000 true accuracy at generation 100 orders CLEAN > MIXED > LEAK, and LEAK's true gain is less than half of CLEAN's.",
     "True accuracy was 0.776 (CLEAN), 0.732 (MIXED) and 0.696 (LEAK); CLEAN's gain was +0.062 and LEAK's -0.017: passed. Training on failed reused items lowered true accuracy."),
    ("P5", ["P5. LEAK, n = 16000: phantom share of reported gain over generations", "51-100 exceeds 0.7."], '"P5":{"phantomLate":0.763,"pass":true}',
     "For LEAK at n = 16000 the phantom share of reported gain over generations 51-100 exceeds 0.7.",
     "The phantom share was 0.763: passed."),
    ("P6", ["P6. LEAK, n = 16000: reported gain over generations 51-100 exceeds 0.02", "while true gain over the same span is below 0.01."], '"P6":{"repLate":0.01,"truLate":0.002,"pass":false}',
     "For LEAK at n = 16000 the reported gain over generations 51-100 exceeds 0.02 while the true gain over the same span is below 0.01.",
     "The reported late gain was 0.01 (below the 0.02 floor) and the true late gain 0.002: failed on the reported-gain half (0.01 against the 0.02 floor)."),
]
for _k, _pq, _rq, _pred, _out in RUN2:
    R(EL, "learning_gain_prediction", "observed", EM, EP2, "Run 2 prediction " + _k,
      _pq + [(ER2, _rq)],
      CTX_E2, SK_EM,
      "Run the sealed item-level-feedback simulation and score registered prediction %s against its range." % _k,
      "Registered before the first run (hashes in SEAL_2.txt sealed first): " + _pred,
      _out,
      "Compare the computed %s statistic with the registered range or ordering; report failures as failures." % _k,
      "preregistered")

R(EL, "failure_diagnosis", "observed", EM, "output/eval_reuse_toys/README.md", "Prediction record",
  ["Prediction record: 5 of 6 passed in run 1, 3 of 6 in run 2. Both runs\noverpredicted the size of the gap between reported and true accuracy."],
  "Two sealed toy runs predicted how far the score on a reused held-out set separates from true accuracy. Run 1 had six registered predictions (score-only feedback) and run 2 had six (item-level feedback with a 3-NN learner).",
  SK_EM,
  "Compare the registered predictions with the outcomes across both runs and look for a shared direction in the misses.",
  None,
  "Five of six predictions passed in run 1 and three of six in run 2. Both runs overpredicted the size of the gap between reported and true accuracy: for example the naive gap at n = 4000 was 0.035 against a registered [0.08, 0.25], and the leak gap at n = 16000 was 0.015 against [0.025, 0.065].",
  "Tabulate each registered range against its outcome and calibrate the next prediction's range on the direction of the misses.",
  "none")

R(EL, "experiment_design", "observed", EM, EP2, "Why a second design; README",
  ["Run 1 used a parametric toy with score-only feedback.", "Here the learner is real, so that share is measured.",
   (ER2.replace("results2.json", "README.md"), "Post hoc arm added after run 2's results were seen. Exploratory."),
   (ER2.replace("results2.json", "README.md"), "reproduced all three results files byte for byte")],
  "Run 1 of the reuse toy used a parametric model with score-only feedback. Item-level feedback (the loop seeing which items failed) is the mechanism of interest, but in the parametric model the share of a patch that generalizes would have to be an assumed parameter.",
  SK_EM + ["3-nearest-neighbour real learner", "post hoc exploratory arm (sim2b)"],
  "Replace the parametric learner with a real k-NN learner so that how much of a patch generalizes is measured rather than assumed, register the predictions and seal the files first, and label any arm added afterwards as post hoc.",
  None,
  "Run 2 passed 3 of 6 predictions. A post hoc arm (sim2b.js) was added after run 2's results were seen and is labelled exploratory, with hashes appended afterwards. A re-run from a scratch copy on the same day reproduced all three results files byte for byte; on Node 24.19.0 under Windows two of three reproduced exactly and results.json differed in the last digit of 24 floats (largest relative difference about 1.2e-15), so run 1 is reproducible to its reported precision, not to the byte.",
  "Seal hashes before running, rerun from a scratch copy, and report cross-platform differences with their size.",
  "none")

# ---------------------------------------------------------------------------
# StableRateStudy
# ---------------------------------------------------------------------------
SR = "srs"
SL = "stable-rate-study"
R(SL, "learning_gain_prediction", "observed", SR, "REPORT.md", "Verdict",
  ["By the pre-registered rules, H_stable is supported and H_fast is not.",
   "at most 0.9 bits of extra precision per doubling of W, against 9.4 for the bit-extraction network and 2.05 for the stable interpolant network",
   (  "PREREGISTRATION.md", "H_stable: trained networks show an exponent gamma <= 1"), ("PREREGISTRATION.md", "H_fast: deep networks at high precision show gamma > 1, with sensitivity growing")],
  "Do trained ReLU networks beat the stable approximation rate? Random Hoelder targets on [0,1]^d (d = 1, s = 0.5; d = 2, s = 0.8); plain ReLU MLPs of depth 1, 3 and 8 and a residual arm whose depth grows with W; budgets W = 100 to 31,623; float32, QAT at 16, 8 and 4 bits, and float64; 4 seeds each, 652 trained members. Hypotheses were fixed before any network was trained on the main targets.",
  ["Adam with plateau rule and fresh uniform batches", "exponent gamma_eff from effective interpolation cells", "weight-sensitivity ladder and post-training rounding", "hand-built bit-extraction control"],
  "Test H_stable (trained exponent gamma <= 1) against H_fast (deep, high-precision networks show gamma > 1 with growing precision sensitivity) using pre-registered decision rules and a hand-built bit-extraction control.",
  "Registered hypotheses: H_stable gamma <= 1; H_fast gamma > 1 with sensitivity growing. The decision rules were written before training.",
  "H_stable was supported and H_fast was not: no trained family had an exponent significantly above the stable one, float-trained families gained at most 0.9 bits of precision need per doubling of W against 9.4 for the bit-extraction network and 2.05 for the stable interpolant network, and the hand-built control showed the fast exponent is representable but its packed bits receive zero gradient.",
  "Apply the registered verdict rule per family and report seeds, bootstrap intervals and budget checks, including those that fail.",
  "preregistered")

R(SL, "failure_diagnosis", "observed", SR, "REPORT.md", "Verdict, qualification 1",
  ["In d = 1, training does not even reach the stable rate.", "a 31,000-weight depth-3 network is as accurate as an interpolant using about 1% of its weights",
   "Their exponent is about 0.5-0.7 of\n   the stable one"],
  "Trained ReLU networks in d = 1 are compared with a piecewise-linear interpolant that uses the same number of weights.",
  ["interpolant written as a one-hidden-layer ReLU network", "basin probe training both from perturbed starts", "effective interpolation-cell count N_eff"],
  "Compare the trained network with an interpolant at the same weight budget and with the interpolant's own exponent before interpreting a below-one exponent as a rate result.",
  None,
  "Trained networks fall further behind the stable reference as W grows, with exponents about 0.5-0.7 of the stable one; a 31,000-weight depth-3 network is as accurate as an interpolant using about 1% of its weights. 'Did not beat the stable rate' therefore largely means 'did not get close to it', and d = 1 says little about whether a better optimiser could find faster solutions.",
  "State how far the trained network is from the reference, not just that the registered inequality holds.",
  "none")

R(SL, "quantization_aware_adaptation", "observed", SR, "REPORT.md", "Verdict and Results 1",
  ["No family collapses under weight rounding: trained networks stay within 2x of\n  their error with at most 9 significant bits per weight", "rounding to 8 bits costs at most 1.15x",
   "0.51 [0.26, 0.75]** | – | 0.54 | 0.27 | -0.10 | –"],
  "Trained ReLU networks are evaluated in float32 and with quantization-aware training (per-tensor fixed point on every parameter, straight-through gradients) at 16, 8 and 4 bits, and their weights are rounded after training.",
  ["QAT at 16, 8 and 4 bits", "post-training rounding", "effective exponent gamma_eff", "bits needed = precision at which a perturbation matches the network's own error"],
  "Train at several QAT bit widths and also round weights after training, and read both the exponent and the bits needed.",
  None,
  "No family collapsed under weight rounding: trained networks stayed within 2x of their error with at most 9 significant bits per weight (that number grows by at most 0.6 bits per doubling of W), and rounding to 8 bits cost at most 1.15x for the deep d = 2 families. QAT lowered the fitted exponent: for d = 1 depth 3 it was 0.51 in float32, 0.54 at 16 bits, 0.27 at 8 bits and -0.10 at 4 bits.",
  "Compare float32, QAT and post-training rounding on the same members, with seeds sharing initialisation and data stream across precisions.",
  "none")

R(SL, "experiment_design", "observed", SR, "REPORT.md", "Verdict, qualification 2",
  ["An exploratory fourth point at W = 31,745 shows\n   that lead levelling off (local exponent 1.46, 1.17, then 1.06).", "All intervals\n   include 1.",
   "The depth-8 exponent\n   fails the budget check (it rose 0.23 when training was doubled)."],
  "In d = 2, deep arms sit near the stable curve and some point estimates exceed 1 (residual arm 1.28 sup and 1.31 L2 from three W values; depth 8 after doubling the training budget 1.09 [0.68, 1.50]).",
  ["extra W point added after seeing data (logged as an exploratory change)", "budget check that doubles training", "bootstrap intervals over seeds"],
  "Add a fourth W point to the residual arm as a clearly labelled exploratory run, and check budget sensitivity by doubling the training budget.",
  None,
  "The exploratory fourth point at W = 31,745 showed the lead levelling off (local exponent 1.46, 1.17, then 1.06). All intervals include 1, and the depth-8 exponent failed the budget check, rising 0.23 when training was doubled, so a modestly super-stable exponent from better optimisation of larger networks cannot be excluded.",
  "Log every post-hoc addition in the pre-registration's change section with time and reason.",
  "none")

# ---------------------------------------------------------------------------
# loop-capacity
# ---------------------------------------------------------------------------
LC = "lc"
R("loop-capacity/mod7-quantized-reach", "quantization_aware_adaptation", "observed", LC, "README.md", "Results at a glance",
  ["Quantized models trained at T = 32 learn a scan\n  whose reach keeps growing with test-time passes.", "34.5 at 32 passes to 61 at 64"],
  "Weight-tied transformers trained on a running-sum-mod-7 scan with bf16, int4 or ternary weights (per-model fake quantization with a straight-through estimator) are evaluated with more test-time passes than they were trained on. n* is the largest instance size solved at accuracy 0.9.",
  ["weight-tied block run T times", "per-model fake quantization (float, int4, ternary)", "test-time pass extrapolation"],
  "Train the quantized scan models at T = 32 passes and evaluate at 32 and 64 test-time passes.",
  None,
  "Quantized models trained at T = 32 learned a scan whose reach kept growing with test-time passes; a ternary model went from n* = 34.5 at 32 passes to 61 at 64.",
  "Report n* at each test-time pass count, not only at the training T.",
  "none")

R("loop-capacity/phase1-grid", "experiment_design", "observed", LC, "README.md", "Results at a glance",
  ["Tied beats untied at equal parameters once T ≥ 8.", "It loses to untied at equal compute: 12.5 vs 23.5 at T = 32.", "W^0.24 · (W·T)^0.18 · b^0.26 on rule 110",
   "Pre-registered verdict: mixed for both families."],
  "A pre-registered grid asks whether a weight-tied network that runs more passes can solve larger fixed-rule instances, or whether size is capped by weight bits W times b. Arms: tied, untied with matched parameters (untied_w), and untied with matched compute (untied_c); tasks: rule-110 diagonal and running sum mod 7; bit widths bf16, int4 and ternary; T from 1 to 32.",
  ["tied, untied-matched-W and untied-matched-compute arms", "n* at accuracy 0.9", "pre-registered decision rules with dated amendments"],
  "Compare tied against untied at both equal parameters and equal compute, since the verdict flips with the matching rule.",
  None,
  "Pre-registered verdict: mixed for both families. Tied beat untied at equal parameters once T >= 8 but lost at equal compute (n* 12.5 vs 23.5 at T = 32); across arms n* scaled as W^0.24 (W T)^0.18 b^0.26 on rule 110.",
  "Report each comparison under both matching rules.",
  "none")

R("loop-capacity/phase2-storage", "learning_gain_prediction", "observed", LC, "README.md", "Phase 2",
  ["Measured directly, storage is flat in passes: tied models memorizing\n    random labels store about 1.8 bits per parameter in float, against a\n    16-bit bound, from T = 4 to T = 32.",
   "Passes buy computation, not storage.", (  "results/phase2/summary.md", "| 16 | float (bf16) | 3,773 (1.22) | 5,106 (1.65) | 5,449 (1.76) | 5,617 (1.81) | 5,116 (1.65) | 5,661 (1.82) | 49,666 | flat (1.04) |")],
  "Does running a weight-tied network more times raise how many random label bits it can memorize? Tied models of width d = 16 and 32 in bf16, int4 and ternary memorize random labels at T = 1 to 32 passes; the counting bound is W b + scale bits + log2 T.",
  ["random-label memorization (memrand design)", "stored-bits estimate", "counting-bound check"],
  "Measure storage directly by memorizing random labels at several pass counts, and check the counting bound on every run.",
  None,
  "Storage was flat in passes: for d = 16 in bf16 stored bits were 3,773 at T = 1 and about 5,100 to 5,700 from T = 2 to T = 32 (about 1.8 bits per parameter against a 49,666-bit bound); int4 and ternary also flat (about 0.4 and 0.13 bits per parameter). There were zero bound violations. Passes buy computation, not storage.",
  "Check for saturation (last two logged values within 5%) and report how many models were unsaturated (25 of 54 here).",
  "none")

R("loop-capacity/phase2-storage", "failure_diagnosis", "observed", LC, "results/phase2/summary.md", "Q1 (mem16, original design)",
  ["## Q1 (mem16 (original design, no signal; see amendment 1)): memorization capacity against passes", "| 16 | float (bf16) | 0 (0.00) | -1 (-0.00) | -4 (-0.00) |",
   "Unsaturated models (last two logged values differ by > 5%): 54 of 54."],
  "The original memorization design (mem16) was run at d = 16 and 32 across bit widths and pass counts, and stored-bit estimates came back at zero or slightly negative for every cell.",
  ["stored-bits estimate with a no-signal flag", "unsaturated-model count", "dated pre-registration amendment"],
  "Treat an all-zero stored-bits table as a design failure (no signal), keep the table, and amend the design in a dated amendment before rerunning with a task that can be learned (memrand).",
  None,
  "Every cell stored about 0 bits (for example -1 at T = 2 and -15 at T = 32 for d = 16 bf16) and all 54 of 54 models were unsaturated: the verdict was 'no signal', and the design was amended (amendment 1) to the memrand task, which produced a signal in the same table.",
  "Look at the unsaturated count and the sign of stored bits before interpreting a capacity-against-passes curve.",
  "none")

TX = "loop-capacity/test-time-extrapolation"
R(TX, "transfer_across_families", "observed", LC, "results/phase3/summary.md", "Phase 3 summary",
  ["no clear effect (1.16)", "hurts (0.50)", (  "README.md", "It does not rescue ternary models or rule 110."), (  "README.md", "No arm meets the pre-registered 2× extrapolation rule."),
   (  "README.md", "stable out to 512 passes")],
  "Looped scan models are trained at one pass count and evaluated at many more (up to 512) and on larger instance sizes than training covered. Phase 3 adds a progressive depth-stability loss (a baseline is the same recipe without that term).",
  ["progressive depth-stability loss", "n* by test-time passes", "2x extrapolation decision rule", "drift and horizon diagnostics"],
  "Add the progressive depth-stability loss to the scan training and judge it by extrapolation to 128 to 512 test-time passes against the baseline recipe.",
  "Registered rule: an arm extrapolates only if it meets the pre-registered 2x extrapolation criterion against the baseline recipe at the larger pass counts.",
  "The loss made float local-attention scans stable out to 512 passes, but no arm met the pre-registered 2x extrapolation rule: the effect at 128 passes was 'no clear effect' (ratio 1.16) for mod-7 local attention, and for the rule-110 local arm it hurt (ratio 0.50). It did not rescue ternary models or rule 110.",
  "Judge extrapolation by n* ratio at the largest pass count, with a baseline of the same recipe.",
  "preregistered")

R(TX, "transfer_across_families", "observed", LC, "results/phase2/summary.md", "Q2 extrapolation in test-time passes",
  ["| extrap_mod7_tied_d32_T32_N512 | 16: 16, 32: 25, 64: 25, 128: 18, 256: 2, 512: 0.5 | 0.72 | plateaus |",
   "| extrap_mod7_tied_d32_T16_N512 | 16: 10.5, 32: 10, 64: 4, 128: 0, 256: 0, 512: 0 | 0.00 | plateaus |",
   (  "README.md", "Serial reach grows with passes only out to the problem sizes training\n    covered.")],
  "A tied d = 32 running-sum-mod-7 model trained at T = 16 or T = 32 passes is evaluated at 16 to 512 test-time passes. n* is the largest instance size solved at accuracy 0.9; the extrapolation ratio is n*(128)/n*(32).",
  ["tied weight-sharing", "test-time pass sweep", "extrapolation ratio n*(128)/n*(32)"],
  "Evaluate each trained model at pass counts far beyond its training T and report the ratio of reach at 128 passes to reach at 32.",
  None,
  "Reach plateaued and then collapsed: trained at T = 32, n* was 16, 25, 25, 18, 2 and 0.5 at 16, 32, 64, 128, 256 and 512 passes (ratio 0.72); trained at T = 16 it fell to 0 by 128 passes (ratio 0.00). Serial reach grew with passes only out to the problem sizes training covered.",
  "Report reach at every test-time pass count and mark the verdict 'plateaus' when it stops growing.",
  "none")

# ---------------------------------------------------------------------------
# jev-qwen xbench
# ---------------------------------------------------------------------------
JQ = "jq"
JP = "reports/xbench/summary.md"
R("jev-qwen/xbench-depth-extrapolation", "transfer_across_families", "observed", JQ, JP, "RQ-K: compactification, hidden depth vs emitted reasoning",
  ["Within trained depths, the 460k-param looped decoder is best (LOOP-T-ds 0.999 on depths 1-8).", "Only emitted reasoning extrapolates: 0.717 on depths 9-12, against 0.308 for LOOP-T-ds.",
   "It does not help past depth 8.", "A trained loop on the task rows adds hops (about 6)"],
  "Pointer-chasing depth suite S1 in a pre-registered cross-architecture benchmark (xbench): a 460k-parameter looped decoder (LOOP-T-ds), emitted-reasoning arms, and a 4B Qwen3.5-based Jev-style model with and without a trained loop. Training covers depths 1 to 8; test depths 9 to 12 are unseen.",
  ["looped decoder with weight-shared transitions", "emitted chain-of-thought reasoning", "layer-loop driver on Qwen3.5-4B", "paired training seeds"],
  "Choose the architecture by whether the test depth falls inside the trained depths: a looped decoder inside the trained range, emitted reasoning beyond it.",
  None,
  "Within trained depths the 460k-parameter looped decoder was best (LOOP-T-ds 0.999 on depths 1-8). On unseen depths 9-12 only emitted reasoning extrapolated: 0.717 against 0.308 for LOOP-T-ds. A 4B single pass tracked 1-2 hops untrained and 3-4 after training on the task rows; a trained loop on the task rows added about 6 hops and beat loop-free training in every seed (pooled one-sided p = 1.3e-8) but did not help past depth 8.",
  "Score trained-depth and unseen-depth ranges separately.",
  "none")

R("jev-qwen/xbench-depth-extrapolation", "transfer_across_families", "observed", JQ, JP, "RQ-K: compactification, mixed-suite loop",
  ["A loop trained on the mixed decision suite adds nothing (H4 falsified in all three seeds)."],
  "The same Qwen3.5-4B layer-loop is trained on a mixed decision suite instead of on the depth task rows, and evaluated on algorithmic depth. Hypothesis H4 was registered: loop training on the mixed suite adds hops.",
  ["layer-loop driver", "mixed decision-suite training", "three training seeds"],
  "Train the loop on the mixed decision suite and test whether it transfers to the depth suite.",
  "Registered hypothesis H4: loop training on the mixed decision suite adds depth capability.",
  "H4 was falsified in all three seeds: a loop trained on the mixed decision suite added nothing on algorithmic depth, whereas a loop trained on the depth task rows added about 6 hops.",
  "Use three training seeds and report the hypothesis outcome per seed.",
  "preregistered")

R("jev-qwen/xbench-s7-routing", "specialist_routing", "observed", JQ, JP, "RQ-K: smallest reliable executor",
  ["| S7 skill routing | lexical router | SkillRouter-TRM (6,338) |", "Smallest reliable executor (registered rule, registered seed-0 arms)"],
  "Suite S7 of the xbench asks which executor routes prompts to skills reliably, under a registered smallest-reliable-executor rule applied separately to all arms and to neural arms only. Arms include script and lexical routers, a 6,338-parameter TRM skill router and 4B Jev-style models.",
  ["lexical router", "SkillRouter-TRM (6,338 parameters)", "script and kNN gates", "registered reliability rule"],
  "Apply the registered rule to find the smallest executor that meets it, in the all-arms pool and in the neural-only pool.",
  None,
  "On skill routing the smallest reliable executor was the lexical router among all arms and SkillRouter-TRM (6,338 parameters) among neural arms.",
  "State the rule and the pool (all arms or neural only) next to every smallest-executor claim.",
  "none")

R("jev-qwen/xbench-control", "failure_diagnosis", "observed", JQ, JP, "RQ-C: control, monitor ladder",
  ["J-V0, J-V1 and J-V1c let 1 of 10 attacks through with at most 1 false block in 22.", "Every J monitor misses the same duplicate-lineage quorum attack, with p(ALLOW) 0.94-0.999."],
  "Suite S5 scores monitors on attack and benign items, including a duplicate-lineage quorum attack. Monitors: the Jev-style Qwen3.5-4B arms, an action gate, a trajectory budget, a delegation guard, an ignore-every-claim baseline and a provenance membrane.",
  ["Jev-style Qwen3.5-4B monitors (J-V0, J-V1, J-V1c)", "action gate", "trajectory budget", "delegation guard", "provenance membrane"],
  "Inspect which attack each monitor misses, not only the aggregate pass count, because the misses can all be the same item.",
  None,
  "J-V0, J-V1 and J-V1c let 1 of 10 attacks through with at most 1 false block in 22, beating the action gate (0.80), trajectory budget (0.60) and delegation guard (0.50). Every J monitor missed the same duplicate-lineage quorum attack, with p(ALLOW) between 0.94 and 0.999.",
  "List the missed attack identities per monitor.",
  "none")

R("jev-qwen/xbench-control", "skill_composition", "observed", JQ, JP, "RQ-C: control, monitor ladder",
  ["Ignoring every claim stops all attacks at 23% false blocks. The provenance membrane (q4)\n    stops all with none.", "Bonsai-8B lets 3 of 10 through, with 4 false blocks."],
  "In the S5 monitor ladder, monitors see attack and benign items that carry claims. Options: a language-model monitor, a rule that ignores every claim, or the provenance membrane (q4).",
  ["provenance membrane (q4)", "ignore-every-claim baseline", "Bonsai-8B as a monitor"],
  "Add the provenance membrane (q4) to the monitor stack instead of relying on a language-model monitor alone.",
  None,
  "Ignoring every claim stopped all attacks at 23% false blocks, while the provenance membrane stopped all attacks with no false blocks; Bonsai-8B let 3 of 10 attacks through with 4 false blocks.",
  "Report attacks stopped and false blocks together for every monitor.",
  "none")

R("jev-qwen/xbench-compactification", "experiment_design", "observed", JQ, JP, "Completion checklist and RQ-K",
  ["S2 commit/veto | none: underpowered (18 true rejects) | none", "S5 monitor ladder | none: underpowered (11 attacks) | none",
   "No Jev-style arm and no Bonsai arm is reliable on any suite, in either pool.", "SPEC registered before outcomes; changes only as dated addenda"],
  "A cross-architecture benchmark with seven suites compares Jev-style 4B arms, compact looped decoders, TRM skill gates, script and kNN gates and a 1-bit 8B model. The specification was registered before any cross-arm outcome and changed only through dated addenda (two post hoc).",
  ["registered smallest-reliable-executor rule", "Wilson 95% intervals", "Holm-adjusted paired tests", "coverage check of every (arm, suite) cell"],
  "Apply the registered rule per suite and report 'none: underpowered' where the suite has too few true rejects or attacks, rather than naming a winner.",
  None,
  "The rule gave 'none: underpowered' on S2 (18 true rejects) and S5 (11 attacks). No Jev-style arm and no Bonsai arm was reliable on any suite in either pool, and on the log10(params x passes) axis the 4B arms sit four to six orders of magnitude to the right of the reliable executors. All 189 (arm, suite) cells have a record or a written N/A reason.",
  "Run the coverage check so no cell is missing and report underpowered suites as underpowered.",
  "none")

# ---------------------------------------------------------------------------
# ConstitutionalAlignment
# ---------------------------------------------------------------------------
CA = "ca"
CAD = "experiments/frame_internalization_sft_v1/local_screen_v1/LOCAL_SCREEN_RESULT_V1.md"
R("constitutional-alignment/qwen3-1p7b-worldview-screen", "learning_gain_prediction", "observed", CA, CAD, "Frozen-screen outcomes",
  ["The prospectively frozen local screen completed on 2026-07-21.", "Logged loss: 6.1609 at step 1 and 3.7473 at step 30.",
   "| No-cue preferred-proxy rate | 0.833 | 0.750 | -0.083 |", "| All-variant action stability | 0.167 | 0.250 | +0.083 |",
   "The flavored screen failed because marker uptake did not increase. The guided screen failed because the preferred-proxy delta was negative."],
  "A 30-step QLoRA (rank 8, 512-token sequences, 238 training rows, 40 validation rows, 3,211,264 trainable parameters) on exact local Qwen3-1.7B on a 4 GB RTX 3050 Laptop GPU, trained on a MeTTa worldview curriculum. 56 frozen no-frame probes are scored for the base and the adapter with no retries or repairs.",
  ["QLoRA rank 8 on a 4 GB GPU", "MeTTa skill/value graph for preferred-action proxies", "frozen no-frame probe set", "prospectively frozen pass rules"],
  "Run the cheap frozen local screen with predeclared pass rules (valid action rate above 0.95, marker uptake up, preferred-proxy delta positive) before paying for a larger model.",
  "Predeclared rules: the infrastructure screen passes if training completes and valid action rate exceeds 0.95; the flavored screen needs marker uptake to rise; the guided screen needs a positive preferred-proxy delta.",
  "The infrastructure screen passed (30/30 steps, loss 6.1609 to 3.7473, adapter valid action rate 0.982). Both judgment screens failed: marker uptake did not rise and the preferred-proxy rate moved from 0.833 to 0.750 (-0.083), with all-variant action stability 0.167 to 0.250 and opposite-pressure preferred-proxy rate 0.000 to 0.083. Worldview-native reasoning was not tested.",
  "Apply the frozen rules exactly as written and report the negative result for the exact intervention.",
  "preregistered")

R("constitutional-alignment/qwen3-1p7b-worldview-screen", "failure_diagnosis", "observed", CA, CAD, "Interpretation and next decision",
  ["This is a negative result for the exact 30-step 1.7B intervention, not a general verdict against 1.7B worldview conditioning.",
   "lacks a neutral SFT control", "Its preferred actions are evaluation proxies derived from the MeTTa skill/value graph, not normative ground truth.",
   "The only no-cue action switch occurred on `harbor_granary_dev:allocation:no_cue`."],
  "The 30-step 1.7B worldview adapter failed both judgment screens. The run used development cases only, covered about 0.126 epochs and 5,499 tokens, and had no neutral-SFT control arm.",
  ["frozen probe outputs for base and adapter", "per-case action-switch table"],
  "Look at the individual action switches and the missing controls before attributing the failure to the model size or the worldview content.",
  None,
  "The only no-cue action switch was on harbor_granary_dev:allocation:no_cue, where the base chose transparent need-based allocation and the adapter chose quiet merchant favoritism, away from the MeTTa-derived proxy. The source treats the result as negative for this exact intervention only: it was cheap and short, used development cases, lacked a neutral SFT control, and its preferred actions are proxies, so it does not justify a 4B or 9B run yet; the next move is a denser supervision intervention at 1.7B under a new prospective amendment.",
  "Repeat under a new prospective amendment with a neutral SFT control and denser skill-to-decision coverage.",
  "none")

R("constitutional-alignment/hrm-v1-smoke", "transfer_across_families", "observed", CA, "experiments/constitutional_hrm_v1/RESULTS_20260721.md", "whole file",
  ["| Constitutional | 0.8333 | 0.5625 | 0.6452 |", "| Unweighted utility | 0.5625 | 0.1875 | 0.0968 |", "| Shuffled labels | 0.5208 | 0.4583 | 0.5000 |",
   "OOD\npeaked at 0.6667 on step 50 and then fell to 0.5625", "The constitutional contrast\ndelta over shuffled was +0.1452, which passed its +0.10 gate."],
  "A bounded local smoke trains a small constitutional HRM model with three arms (constitutional, unweighted utility, shuffled labels) and scores in-distribution, out-of-distribution (compositional) and disagreement-slice contrast accuracy against frozen gates. All arms start at exactly 0.50.",
  ["constitutional-weighted training arm", "unweighted-utility control", "shuffled-label control", "frozen promotion gate G5 with contrast threshold 0.75"],
  "Judge the constitutional arm on the held-out composition slice and on the disagreement contrast, not on in-distribution accuracy alone.",
  "Frozen gates: a +0.10 contrast delta over shuffled and an absolute contrast accuracy of 0.75.",
  "In-distribution accuracy was 0.8333 (constitutional), 0.5625 (unweighted) and 0.5208 (shuffled); out-of-distribution 0.5625, 0.1875 and 0.4583; contrast 0.6452, 0.0968 and 0.5000. The +0.10 delta gate over shuffled passed (+0.1452) but the 0.75 absolute contrast gate failed, and out-of-distribution accuracy peaked at 0.6667 at step 50 before falling to 0.5625, so more steps would not help and promotion gate G5 was not cleared.",
  "Report all three slices per arm with the controls and the checkpoint curve.",
  "preregistered")

R("constitutional-alignment/frame-internalization-contrast", "experiment_design", "hypothesis", CA,
  "experiments/frame_internalization_sft_v1/PROTOCOL_AMENDMENT_DIRECT_PROMPT_SFT_V1.md", "Protocol amendment",
  ["Status: frozen before training or evaluation outcomes; execution pending", "No prompt,\njudge, exclusion, or threshold may change after outcomes become readable.",
   "Missing, duplicate, extra, null, or nonfinite join values stop the analysis.", "Run 10,000 paired prompt-cluster bootstrap draws with seed 42."],
  "A prompt study and an SFT study of whether a model internalizes a framing both exist separately, but their difference is not a single estimand. G is the free-tier minus paid-tier strong-reject compliance gap over 200 frozen harmful prompt IDs.",
  ["prompt-cluster paired bootstrap", "simultaneous 95% intervals over registered contrasts", "frozen judge and parsing rules"],
  "Freeze a direct contrast D = E_SFT - E_prompt on the same prompts, tiers, decoding settings, judge and seeds before any outcome is read, with a bootstrap seed and stop-on-missing-data rules.",
  "Not a numeric prediction: neither sign of D is interpreted unless benign-over-refusal, capability, base-reanchor and evaluator gates pass.",
  "Resample the same prompt IDs across all arms and tiers; stop the analysis on any missing, duplicate or null join value.",
  "preregistered")

R("jev-qwen/xbench-s2-commit-veto-gates", "specialist_routing", "observed", JQ, JP, "RQ-C: control, unsafe allowance and over-refusal",
  ["S2: J arms commit 0-2 of 18 true rejects and refuse 2-4 of 56 true commits. Script, kNN and\n    TRM-cv: 0 and 0. Bonsai-8B commits 8 of 18.",
   "S4 deny items: J arms allow 0 of 54 (1 of 54 for V1c seed 1); Bonsai-8B allows 7 of 54."],
  "Suite S2 (commit/veto) has 18 true rejects and 56 true commits; suite S4 (decision suite) has 54 deny items. A request is handled by a gate: a script, a kNN gate, a small TRM gate (TRM-cv), a 4B Jev-style model arm, or the 1-bit Bonsai-8B.",
  ["script gate", "kNN gate", "TRM-cv gate", "Jev-style Qwen3.5-4B arms (J)", "Bonsai-8B as a gate"],
  "Send commit/veto decisions to the script, kNN or small TRM gate rather than to a general language model, and count both unsafe allowances and over-refusals.",
  None,
  "On S2 the J arms committed 0-2 of 18 true rejects and refused 2-4 of 56 true commits; the script, kNN and TRM-cv gates did 0 and 0; Bonsai-8B committed 8 of 18. On S4 deny items the J arms allowed 0 of 54 (1 of 54 for one V1c seed) and Bonsai-8B allowed 7 of 54.",
  "Report unsafe allowance and over-refusal together, with the number of true rejects, because the suite is underpowered at 18.",
  "none")


# ---------------------------------------------------------------------------
# Synthetic, templated records. Every state below is invented for the template and says so.
# No number in a synthetic record is presented as a measurement; allocations are computed from the stated rule.
# Synthetic records are generated only for families that are not held out.
# ---------------------------------------------------------------------------
_NP = "research/self_introspective_meta_learning_v1/NEURAL_PROTOCOL.md"
_NR = "research/self_introspective_meta_learning_v1/NEURAL_RESULTS.md"
_RM = "research/self_introspective_meta_learning_v1/README.md"


def _make_allocation_scenarios():
    rng = random.Random(SEED + 1)
    weights = {"T0": 0.25, "T1": 0.45, "T4": 0.30}
    budget = 320
    for i in range(12):
        acc = {k: round(rng.uniform(0.05, 0.95), 2) for k in weights}
        gaps = {k: 1.0 - acc[k] for k in acc}
        gsum = sum(gaps.values())
        raw = {k: budget * (0.65 * weights[k] + 0.35 * gaps[k] / gsum) for k in weights}
        alloc = {k: int(math.floor(raw[k])) for k in weights}
        rem = budget - sum(alloc.values())
        for k in sorted(weights, key=lambda k: (-(raw[k] - alloc[k]), k))[:rem]:
            alloc[k] += 1
        floor = {k: int(round(0.65 * weights[k] * budget)) for k in weights}
        worst = max(gaps, key=lambda k: gaps[k])
        ctx = ("HYPOTHETICAL learner state invented for this template (not a measurement). A width-48 tied GRU is trained on three task families: "
               "T0 (eight-state hidden-bit world), T1 (compositions of T0 primitives), T4 (modular addition with shear). Current development accuracies: "
               + ", ".join("%s %.2f" % (k, acc[k]) for k in weights)
               + ". Frozen family weights are T0 0.25, T1 0.45, T4 0.30. The next block is 320 AdamW updates of 32 examples.")
        act = ("Allocate the 320 updates with the guarded rule, 65%% by the frozen family weights and 35%% in proportion to each family's development accuracy gap: "
               "T0 %d, T1 %d, T4 %d updates. The largest gap is %s (accuracy %.2f); the rule still keeps every family at or above its floor of T0 %d, T1 %d, T4 %d."
               % (alloc["T0"], alloc["T1"], alloc["T4"], worst, acc[worst], floor["T0"], floor["T1"], floor["T4"]))
        S("synthetic/guarded-allocation/s%02d" % i, "curriculum_selection", "guarded_allocation",
          (SI, _NP), ["mixes its derived preference 35% with the frozen family distribution 65%", "A naive greedy policy is included as a failure control.", (_NR, "Random selection remains strongest on the primary weighted score.")],
          ctx, ["guarded scheduler (35% gap preference, 65% frozen family distribution)", "random allocation", "fixed 5:9:6 allocation", "per-family development accuracy"],
          act,
          "No measured gain is claimed. Qualitatively, positive coverage of every family should prevent the starvation that unconstrained greedy allocation caused in the SIML-01N study, "
          "where random allocation still scored higher than the guarded arm on the primary endpoint.",
          "Run random and fixed allocations with the same 320 updates and seeds as comparison arms, report per-family held-out accuracy, and check that no family fell below its floor.")


def _make_graph_scenarios():
    order = ["T0", "T1", "T2", "T4", "T3", "T5", "T6", "T7"]
    descr = {"T0": "eight-state hidden-bit world", "T1": "compositions of T0 primitives", "T2": "longer recurrent execution",
             "T3": "affine binary circuits", "T4": "modular addition with shear", "T5": "symbolic Storyworld rules with executable consequences",
             "T6": "routed modules solving conjunctions of learned skills", "T7": "partially observed timed planning via an IRC/Pluto adapter"}
    scen = [
        ([], False, False), (["T0"], False, False), (["T0", "T1"], False, False), (["T0", "T1", "T2"], False, False),
        (["T0", "T1", "T2", "T4"], False, False), (["T0", "T1", "T2", "T4", "T3"], False, False),
        (["T0", "T1", "T2", "T4", "T3"], True, False), (["T0", "T1", "T2", "T4", "T3", "T5"], True, False),
    ]
    for i, (done, scorer, adapter) in enumerate(scen):
        def ready(n):
            if n in done:
                return False
            if n == "T1":
                return "T0" in done
            if n == "T2":
                return "T1" in done
            if n in ("T3", "T4"):
                return "T2" in done
            if n == "T5":
                return scorer
            if n == "T6":
                return "T5" in done and "T2" in done
            if n == "T7":
                return adapter
            return True
        nxt = [n for n in order if ready(n)]
        pick = nxt[0] if nxt else None
        blocked = []
        if "T5" not in done and not scorer:
            blocked.append("T5 (needs an independent symbolic scorer)")
        if "T7" not in done and not adapter:
            blocked.append("T7 (waits for an adapter that preserves its observation and timing contract)")
        ctx = ("HYPOTHETICAL graph status invented for this template. Nodes: " + "; ".join("%s %s" % (n, descr[n]) for n in order[:0] or ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"])
               + ". Completed with measured baselines: " + (", ".join(done) if done else "none") + ". Independent symbolic scorer for T5 available: "
               + ("yes" if scorer else "no") + ". T7 adapter available: " + ("yes" if adapter else "no") + ".")
        if pick is None:
            act = "No node is ready: %s. Do not start work on a blocked node." % ("; ".join(blocked) or "all remaining nodes are blocked")
        else:
            act = ("Work on %s (%s) next. %sT6 would additionally need both T5 and T2." % (pick, descr[pick],
                   ("Blocked for now: " + "; ".join(blocked) + ". ") if blocked else ""))
        S("synthetic/graph-next-node/s%02d" % i, "task_graph_navigation", "graph_next_node",
          (SI, _RM), ["T0 -> T1 -> T2 is the first local path.", "T3 and T4 provide independent capacity diagnostics.", "The T5/T2 -> T6 composition edge is a hypothesis.",
                      "T7 waits for an adapter that preserves its declared observation and timing contract."],
          ctx, ["task-graph node contracts", "prerequisite DAG and transfer edges", "independent symbolic scorer", "per-node measured baseline"],
          act,
          "Not a numeric prediction: the choice follows the proposed graph order, and any edge it relies on stays a hypothesis until a matched control measures it.",
          "Record the node's baseline report before work starts and compare any later transfer claim with a matched alternative-training clone.")


_DESIGN = [
    ("adapter-placement", "an adapter-placement sweep that stores random label bits in gate LoRAs of equal parameter budget at different layers and ranks",
     (AB, FU), [],
     "equal trainable budget W for every placement; Fano estimate of stored bits at the end of training; counting-bound check; rounding-knee measurement; an unsaturated and a saturated item count (M) so both regimes are visible; sharpness tracking when a gradient-descent arm is included",
     "predicted stored-bits ranges per placement, written and pushed before the first run, each labelled as proved, established elsewhere, or conjecture"),
    ("ternary-lora-lr", "a learning-rate sweep for LoRA fine-tuning on a frozen ternary model",
     (BA, BP), [],
     "warmup plus cosine decay at every peak rate; an external held-out set whose prompt strings are disjoint from training and asserted in code; held-out perplexity and per-family greedy exact match; a reload test of the saved adapter in a fresh process; at least three seeds if any rate comparison is claimed",
     "a divergence rule written down in advance (loss and held-out perplexity of the saved adapter against the base), since the trainer has no gradient clipping"),
    ("router-ablation", "an ablation of a router over a library of task adapters",
     (Q1, QP), [],
     "an oracle router, a no-adapter baseline and the best single adapter for everything as reference rows; both the full-prompt and the content-only view; routing accuracy and end-to-end task score reported together; a soft and a hard use of any lattice evidence; a misroute cost table",
     "the expected costly confusion named before the run, and the fallback score on the inputs the router gets wrong"),
    ("curriculum-comparison", "a comparison of curriculum schedulers for a small recurrent student on a task graph",
     (SI, RM), [],
     "a fixed human curriculum, a random schedule, a behavioural learning-progress scheduler, the introspective scheduler and a shuffled-diagnostic control with identical call patterns; matched updates, examples and recurrent operations; paired training seeds; the primary endpoint fixed before the secondary transfer endpoint is read",
     "a failure control (unconstrained greedy) included on purpose, and a guarded variant with positive coverage of every family"),
    ("repair-loop", "a test of whether a model can repair a damaged policy through typed patches",
     (BG, BD + "bgi01f_verified_repair_20261030.md"), [],
     "per-family correct counts with a per-group minimum, not only an aggregate; an exact no-regression verifier on every accepted patch; a trusted external checkpoint and a deterministic oracle as labelled controls; model-initiated and externally substituted repairs recorded separately; raw model outputs retained",
     "a predeclared semantic-competence gate on the instrument, so that the repair result is not interpreted when the model cannot read the facts"),
    ("runtime-parity", "a parity test of a quantized runtime across backends and packings",
     (S0, S0D + "logit_parity/PREREG.md"), [],
     "pinned runtime commit, model and text hashes; a primary criterion (top-1 agreement) with its standard error and the mean, 99.9th-percentile and maximum KL divergence reported without pass/fail; the placement of every pass confirmed from the log; an explicit expectation recorded so a miss is visible",
     "a rule that nothing is rerun with changed settings except through a dated amendment that keeps the as-run results"),
]


def _make_design_scenarios():
    for i, (name, what, basis, _x, controls, extra) in enumerate(_DESIGN):
        S("synthetic/design-checklist/%s" % name, "experiment_design", "design_checklist",
          basis, _DESIGN_QUOTES[name],
          "TEMPLATE. A new experiment is planned: %s. It will run on the shared local machine with one job at a time." % what,
          ["pre-registration with sealed hashes", "dated amendments", "paired seeds and matched budgets", "raw-output retention"],
          "Before the first run, write the question, the arms and the measurements, and include: %s. Also include %s. Add anything learned after seeing results only as a dated, labelled post-hoc amendment." % (controls, extra),
          "Not a numeric prediction: this is a checklist, and any predicted range belongs in the pre-registration file.",
          "Hash the pre-registration and the script before the run, commit them, run once as written, and report failed predictions as failures.")


_DESIGN_QUOTES = {
    "adapter-placement": ["Written and pushed **before** any of these runs, on 2026-10-12", "Each prediction is labelled:", "Results will be added under each prediction without editing it."],
    "ternary-lora-lr": ["Warmup + cosine to 1e-6 at the same peak lr was stable", "The trainer has no gradient clipping option.", "prompt strings disjoint from training (asserted in code)"],
    "router-ablation": ["| oracle (true env's adapter) | 1.000 | 0.693 | 1.000 | 0.693 |", "| no adapter (base) | — | 0.494 | — | 0.494 |", "| best single adapter for everything (medmcqa) | — | 0.513 | — | 0.513 |"],
    "curriculum-comparison": [(_NP, "A naive greedy policy is included as a failure control."), "Start by comparing C and E after an engineering smoke run. Add D before attributing any gain to internal diagnostics."],
    "repair-loop": ["Aggregate accuracy would have concealed that entire safety failure.", "| External checkpoint | 7 | 0 | 0 |"],
    "runtime-parity": ["Written and committed before any logit pass ran.", "A failed pass is recorded as it ran, with its log. Nothing is rerun with changed settings without an amendment."],
}


_make_allocation_scenarios()
_make_graph_scenarios()
_make_design_scenarios()



# ===========================================================================
# Pipeline: verify evidence, render, dedupe, split, write
# ===========================================================================
PB_LABEL = {
    "preregistered": "registered before the run",
    "mechanism_rationale": "rationale stated in the source, no number registered",
    "proposed": "proposal, not a measurement",
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text_file(path):
    """sha256 of a text file with CRLF folded to LF, so autocrlf checkouts hash the same."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def stable_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def check_sources_cover():
    used = {}
    for sp in SPECS:
        used.setdefault(sp["src"], set()).add(sp["path"])
        for q in sp["q"]:
            if isinstance(q, tuple):
                used[sp["src"]].add(q[0])
    for sy in SYNTH:
        s, p = sy["basis"]
        used.setdefault(s, set()).add(p)
        for q in sy["basis_q"]:
            if isinstance(q, tuple):
                used[s].add(q[0])
    problems = []
    for key, src in SOURCES.items():
        have = set(src["files_used"])
        want = used.get(key, set())
        if have != want:
            problems.append("%s: only in config %s; only in specs %s" % (key, sorted(have - want), sorted(want - have)))
    for key in used:
        if key not in SOURCES:
            problems.append("unknown source key %s" % key)
    if problems:
        raise SystemExit("configs/corpus_sources.json does not match the files the builder uses:\n  " + "\n  ".join(problems))


def render_messages(ctx, skills, fam, act, pred, pb, ev, out, ver):
    user = "Task context:\n%s\n\nAvailable skills:\n%s\n\nQuestion: %s" % (
        ctx, "\n".join("- " + s for s in skills), FAMILY_QUESTION[fam])
    lines = ["Proposed action: " + act]
    if pred:
        lines.append("Predicted improvement (%s): %s" % (PB_LABEL.get(pb, pb), pred))
    else:
        lines.append("Predicted improvement: none registered in the source.")
    lines.append("How to verify: " + ver)
    if ev == "observed":
        lines.append("What happened (recorded in the source): " + out)
    elif ev == "hypothesis":
        lines.append("Status: registered or proposed in the source; no result is recorded there.")
    else:
        lines.append("Note: templated example with invented inputs; no measured result is claimed.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user},
            {"role": "assistant", "content": "\n".join(lines)}]


def record_id(lin, fam, idx):
    return "ms01-" + hashlib.sha256(("%s|%s|%d" % (lin, fam, idx)).encode("utf-8")).hexdigest()[:16]


def make_evidence_records():
    out = []
    counters = {}
    for sp in SPECS:
        src = SOURCES[sp["src"]]
        prim = []
        for q in sp["q"]:
            p, t = q if isinstance(q, tuple) else (sp["path"], q)
            a, b = locate(sp["src"], p, t)
            if p == sp["path"]:
                prim.append((a, b))
        if not prim:
            raise SystemExit("spec %s/%s has no quote in its primary path %s" % (sp["lin"], sp["fam"], sp["path"]))
        if (sp["pb"] == "none") != (sp["pred"] is None):
            raise SystemExit("spec %s/%s: prediction_basis 'none' must go with predicted_improvement null" % (sp["lin"], sp["fam"]))
        lo, hi = min(a for a, _ in prim), max(b for _, b in prim)
        also = sorted({(q[0] if isinstance(q, tuple) else sp["path"]) for q in sp["q"]} - {sp["path"]})
        prov = {"repo": src["repo"], "commit": src["ref_label"], "commit_sha": src["commit"], "path": sp["path"],
                "lines_or_section": "%s; lines %d-%d" % (sp["sec"], lo, hi)}
        if also:
            prov["also_paths"] = also
        k = (sp["lin"], sp["fam"])
        idx = counters.get(k, 0)
        counters[k] = idx + 1
        out.append(dict(
            id=record_id(sp["lin"], sp["fam"], idx), schema_version=SCHEMA_VERSION, family=sp["fam"],
            task_context=sp["ctx"], available_skills=sp["sk"], proposed_action=sp["act"],
            predicted_improvement=sp["pred"], observed_outcome=sp["out"], verification_method=sp["ver"],
            evidence_type=sp["ev"], lineage_id=sp["lin"], provenance=prov,
            messages=render_messages(sp["ctx"], sp["sk"], sp["fam"], sp["act"], sp["pred"], sp["pb"], sp["ev"], sp["out"], sp["ver"]),
            prediction_basis=sp["pb"]))
    return out


def make_synthetic_records():
    out = []
    counters = {}
    for sy in SYNTH:
        s, bpath = sy["basis"]
        src = SOURCES[s]
        for q in sy["basis_q"]:
            p, t = q if isinstance(q, tuple) else (bpath, q)
            locate(s, p, t)
        k = (sy["lin"], sy["fam"])
        idx = counters.get(k, 0)
        counters[k] = idx + 1
        prov = {"repo": "synthetic-template", "commit": "meta_sft/0.1 builder", "commit_sha": "",
                "path": "experiments/bonsai-gym/scripts/build_corpus.py",
                "lines_or_section": "template %s; follows the procedure stated in %s:%s@%s" % (sy["name"], src["repo"], bpath, src["commit"][:12]),
                "basis": {"repo": src["repo"], "commit": src["ref_label"], "commit_sha": src["commit"], "path": bpath}}
        out.append(dict(
            id=record_id(sy["lin"], sy["fam"], idx), schema_version=SCHEMA_VERSION, family=sy["fam"],
            task_context=sy["ctx"], available_skills=sy["sk"], proposed_action=sy["act"],
            predicted_improvement=sy["pred"], observed_outcome=None, verification_method=sy["ver"],
            evidence_type="synthetic", lineage_id=sy["lin"], provenance=prov,
            messages=render_messages(sy["ctx"], sy["sk"], sy["fam"], sy["act"], sy["pred"], "proposed", "synthetic", None, sy["ver"]),
            prediction_basis="proposed"))
    return out


def _tokens(r):
    text = " ".join(str(r.get(k) or "") for k in ("proposed_action", "predicted_improvement", "observed_outcome"))
    return set(re.findall(r"[a-z0-9.]+", text.lower()))


def dedupe(records):
    kept, dropped = [], []
    by_group = {}
    for r in records:
        g = (r["lineage_id"], r["family"])
        tok = _tokens(r)
        dup = False
        for _, other in by_group.get(g, []):
            inter, union = len(tok & other), len(tok | other)
            if union and inter / union >= DEDUP_JACCARD:
                dup = True
                break
        if dup:
            dropped.append(r["id"])
        else:
            by_group.setdefault(g, []).append((r["id"], tok))
            kept.append(r)
    return kept, dropped


def split_records(records):
    held = [r for r in records if r["family"] in HELDOUT_FAMILIES]
    held_lineages = {r["lineage_id"] for r in held}
    quarantined = [r for r in records if r["family"] not in HELDOUT_FAMILIES and r["lineage_id"] in held_lineages]
    rest = [r for r in records if r["family"] not in HELDOUT_FAMILIES and r["lineage_id"] not in held_lineages]
    lineages = sorted({r["lineage_id"] for r in rest})
    ranked = sorted(lineages, key=lambda l: hashlib.sha256(("%d|%s" % (SEED, l)).encode("utf-8")).hexdigest())
    n_val = max(1, int(round(VAL_FRACTION * len(ranked))))
    val_l = set(ranked[:n_val])
    train = [r for r in rest if r["lineage_id"] not in val_l]
    val = [r for r in rest if r["lineage_id"] in val_l]
    key = lambda r: r["id"]
    return {"train": sorted(train, key=key), "val": sorted(val, key=key), "heldout": sorted(held, key=key)}, quarantined


def validate_record(r):
    """Raise AssertionError describing the first schema violation."""
    need = ["id", "schema_version", "family", "task_context", "available_skills", "proposed_action", "predicted_improvement",
            "observed_outcome", "verification_method", "evidence_type", "lineage_id", "provenance", "messages", "prediction_basis"]
    assert set(r) == set(need), "keys differ: %s" % sorted(set(r) ^ set(need))
    assert r["schema_version"] == SCHEMA_VERSION
    assert r["family"] in FAMILIES, r["family"]
    assert r["evidence_type"] in EVIDENCE_TYPES
    assert r["prediction_basis"] in PREDICTION_BASES
    for k in ("id", "task_context", "proposed_action", "verification_method", "lineage_id"):
        assert isinstance(r[k], str) and r[k].strip(), k
    assert r["id"].startswith("ms01-")
    assert isinstance(r["available_skills"], list) and r["available_skills"] and all(isinstance(s, str) and s for s in r["available_skills"])
    for k in ("predicted_improvement", "observed_outcome"):
        assert r[k] is None or (isinstance(r[k], str) and r[k].strip()), k
    if r["evidence_type"] == "observed":
        assert r["observed_outcome"], "observed record without observed_outcome"
    else:
        assert r["observed_outcome"] is None, "only observed records carry an outcome"
    assert (r["prediction_basis"] == "none") == (r["predicted_improvement"] is None)
    prov = r["provenance"]
    for k in ("repo", "commit", "path", "lines_or_section"):
        assert isinstance(prov.get(k), str) and prov[k], "provenance." + k
    m = r["messages"]
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert all(isinstance(x["content"], str) and x["content"] for x in m)
    assert r["task_context"] in m[1]["content"]
    assert all(("- " + s) in m[1]["content"] for s in r["available_skills"])
    assert r["proposed_action"] in m[2]["content"] and r["verification_method"] in m[2]["content"]
    if r["evidence_type"] == "observed":
        assert r["observed_outcome"] in m[2]["content"]
    else:
        assert "What happened" not in m[2]["content"]
    return True


def counts(records):
    def tally(keyf):
        d = {}
        for r in records:
            d[keyf(r)] = d.get(keyf(r), 0) + 1
        return dict(sorted(d.items()))
    return {"records": len(records), "by_family": tally(lambda r: r["family"]), "by_evidence_type": tally(lambda r: r["evidence_type"]),
            "by_family_and_evidence_type": tally(lambda r: r["family"] + "/" + r["evidence_type"]),
            "lineages": len({r["lineage_id"] for r in records})}


def build_all():
    check_sources_cover()
    records = make_evidence_records() + make_synthetic_records()
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate record ids")
    for r in records:
        validate_record(r)
    n_before = len(records)
    records, dropped = dedupe(records)
    splits, quarantined = split_records(records)
    total_kept = sum(len(v) for v in splits.values())
    n_synth = sum(1 for v in splits.values() for r in v if r["evidence_type"] == "synthetic")
    if n_synth > MAX_SYNTHETIC_FRACTION * total_kept:
        raise SystemExit("synthetic fraction %.3f exceeds %.2f" % (n_synth / total_kept, MAX_SYNTHETIC_FRACTION))
    return splits, quarantined, dropped, n_before


def write_outputs(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits, quarantined, dropped, n_before = build_all()
    files = {}
    for name in ("train", "val", "heldout"):
        data = "".join(stable_json(r) + "\n" for r in splits[name]).encode("utf-8")
        (out_dir / (name + ".jsonl")).write_bytes(data)
        files[name + ".jsonl"] = {"sha256": sha256_bytes(data), "records": len(splits[name]), "bytes": len(data)}
    allrec = [r for v in splits.values() for r in v]
    synth_n = sum(1 for r in allrec if r["evidence_type"] == "synthetic")
    manifest = {
        "dataset": "meta_sft_v0_1", "schema_version": SCHEMA_VERSION, "seed": SEED, "val_fraction": VAL_FRACTION,
        "split_rule": "records of heldout families go to heldout; lineages that appear in heldout are removed from train/val (quarantined); "
                      "remaining lineages ranked by sha256(seed|lineage_id) and the first round(0.10 * n) go to val, the rest to train",
        "heldout_families": HELDOUT_FAMILIES,
        "heldout_lineages": sorted({r["lineage_id"] for r in splits["heldout"]}),
        "builder": {"path": "experiments/bonsai-gym/scripts/build_corpus.py", "sha256": sha256_text_file(Path(__file__).resolve())},
        "sources_config": {"path": "experiments/bonsai-gym/configs/corpus_sources.json", "sha256": sha256_text_file(CONFIG)},
        "sources": [{"key": s["key"], "repo": s["repo"], "commit": s["commit"], "ref_label": s["ref_label"], "files_used": s["files_used"]} for s in SOURCES.values()],
        "files": files,
        "counts": {
            "total": counts(allrec),
            "train": counts(splits["train"]), "val": counts(splits["val"]), "heldout": counts(splits["heldout"]),
        },
        "synthetic_fraction": round(synth_n / len(allrec), 4), "max_synthetic_fraction": MAX_SYNTHETIC_FRACTION,
        "built_from_specs": n_before,
        "collapsed_as_near_duplicates": len(dropped), "collapsed_ids": sorted(dropped),
        "quarantined_records": len(quarantined), "quarantined_ids": sorted(r["id"] for r in quarantined),
        "quarantined_lineages": sorted({r["lineage_id"] for r in quarantined}),
        "dedup_jaccard_threshold": DEDUP_JACCARD,
        "gaps": json.loads(CONFIG.read_text(encoding="utf-8")).get("gaps", []),
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    (out_dir / "manifest.json").write_bytes(data)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--check", action="store_true", help="rebuild to a temp dir and compare with --out")
    args = ap.parse_args(argv)
    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            write_outputs(td)
            bad = []
            for name in ("train.jsonl", "val.jsonl", "heldout.jsonl", "manifest.json"):
                a = (Path(td) / name).read_bytes()
                b = (Path(args.out) / name).read_bytes() if (Path(args.out) / name).exists() else None
                if a != b:
                    bad.append(name)
            if bad:
                print("MISMATCH: " + ", ".join(bad))
                return 1
            print("OK: rebuild is byte-identical")
            return 0
    m = write_outputs(args.out)
    print(json.dumps({k: m["counts"][k]["records"] for k in ("total", "train", "val", "heldout")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
