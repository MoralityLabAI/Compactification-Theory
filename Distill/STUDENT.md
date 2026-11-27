# DISTILL-07 / Research Goblin student and native certificate bridge

## Frozen student experiment

Research-program v0.5 remains unchanged. The student is a small tied recurrent
regret ranker with a 49-feature deterministic public-belief input, 10 hidden
units, 10 action outputs, and one or three cell visits. It predicts negative
oracle continuation regret on legal actions; a host legality mask is mandatory.
The compact belief state is reconstructible from the public prior and evidence
ledger. It never includes the actual hidden world, root ID, oracle values or
provider answers. The current model loops within a decision; it is not a learned
persistent-memory system.

Use 24 train, 8 calibration, 8 ordinary, 8 shifted roots with new fixed split
seeds, four sampled public states per project where available, two initializations
(131,137), 18 epochs and the same optimizer update count for flat, tied-recurrent,
and within-state shuffled-label controls. The stored tensor shapes match; the
one-visit arm has 100 inactive recurrent parameters. The shifted family is
absent from training and calibration. Fit only on synthetic oracle labels.
Calibration selects a conservative reuse-first override threshold (or disables
scoring) by root-balanced sampled-state priced regret improvement. It is not
calibration on trajectory utility.

Evaluate all 64 worlds on each evaluation root, at shared per-world logical
work caps 8,192 and 32,768. Charge belief-feature acquisition, scalar MACs,
activation and bias overhead, legal-action and reward scans, and analytic
branch evaluation. Charge the declared 64 world slots even when the host cache
is warm. This is a deterministic metered model of planner work; it is not
hardware FLOPs or CPU cycles. The host supplies the equivalent public sufficient
state. Its construction and environment transitions are excluded for all arms.
Fallback after insufficient work allowance is reuse-first if affordable,
otherwise defer. A gate pays the score cost even when it does not switch.
Report raw utility, utility minus work/4096, work, fallbacks, certification,
false recommendations, root-paired differences against reuse-first and actual
process CPU time spent in the policy. The 25 ms per-world CPU envelope is a
descriptive qualification only, not a preemptive or equal CPU-time budget.
No native or learned-model promotion follows from this small public experiment.

    python -m Distill.student_campaign --out runs/research-student-v1

Weights and reports stay local, synthetic-only and training_admission=false
for the owning DataGoblin pipeline. The campaign performs actual fitting of an
isolated reference student, not fitting on provider responses or modifying
DataGoblin's native model.

## Native COE certificate bridge

The native source-pinned minimum_observer function is used on finite ordinary
observer problems. A code assignment distinguishes every tested edge (upper
bound); a clique of size c forces at least ceil(log2(c)) bits (lower bound).
A certificate is admitted only when the bounds coincide. The independent
verifier checks every edge, clique pair, word range, source pin and digest.
Transfer maps words and clique via a vertex bijection, then checks the target
graph directly. A graph with changed edges can reuse the witnesses if both
still hold. A hash or matching family name alone is insufficient. The bridge
rejects odd-cycle examples when its clique lower bound is insufficient; this
does not mean the native solver is incorrect.

Benchmark 24 planted graph roots, each with two relabelings and a complete-graph
shift. Compare four cold native solves with a cold anchor, two verified
transports and one rejected transport followed by a new cold solve. Record
native calls, success/rejection, exact width parity, and actual process CPU
including the failed attempt. This proves concrete finite mathematical
statements using the existing native verifier, not a Lean kernel theorem or
a native longitudinal controller. The same receipt structure can later admit
proof-term, BPD and other certificates through separately trusted checkers.

    python -m Distill.coe_certificate --out runs/coe-certificate-reuse

No provider calls, paid expenditure, GPU use, or model output admission.
