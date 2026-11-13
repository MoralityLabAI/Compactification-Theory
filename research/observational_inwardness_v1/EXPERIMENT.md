# OIC-11: preregistered parity-latch monitoring experiment

Status: **prospective** (2026-11-10). The finite OIC-01–10 mathematical
reference and Python tests are verified. **No trained recurrent model
has been evaluated for this protocol.** The hypotheses below do not
assert that transformers possess a modular latent phase.

## Aim

Test whether a monitor can preserve evidence of a hidden recurrent mode
using fewer active interventions than a transition-event-only
measurement policy, with all observation samples and computational
resources charged. This is a possible *application* of the
OIC-08/T09 parity-latch construction, and is falsifiable.

## Registered tiers

**Tier E, exact control.** Use the existing Z2×Zn finite model for
n in {3,4,5,6,7,8}, all n initial hidden phases, both model families,
fixed visible x0=0, fixed actions u in {0,1}. This validates the
instrumentation. The event sensor receives b_t=u_t[z_t=0]; the
state sensor receives x0,...,xT.

**Tier L, learned recurrence.** Train a tiny GRU or tied-weight
recurrent block with an explicit n-phase latent target. Train separate
fixed-phase and rotating-phase conditions and match parameter count,
tokens, optimizer steps, seeds, hidden width and rollout compute.
A controller injects an n-step phase-sensitive test pulse.
The monitor reads only a frozen low-dimensional output head, with
network activations and recurrent caches logged for diagnosis. A
linear hidden-phase classifier may be trained on a *separate*
split and must not become part of the monitor's decision interface
without charging its input bits, weights, and calibration data.

**Tier P, pretrained recurrence.** Only after a Tier L replication,
sample a pinned checkpoint from a looped-layer or latent-depth model.
Record whether the forward step is well-defined from the declared
state, including KV cache, positional state, loop counter and
external decoder. Demonstrate separately that two intervention-defined
histories share a baseline visible transcript while differing in
a repeatable hidden activity pattern. Without both properties,
label the pilot *unsuitable for OIC transfer*.

## Independent variables

- Phase family n, with both parities and at least three odd/even matched pairs
- Number q of active probes, time horizon T, and placement of passive tail
- Observation channel: raw-state snapshots / transition-event audit
- Fault type: bounded e substitutions, r marked erasures; secondary
  stochastic Bernoulli errors with a **registered** error rate
- Held-out hidden initial phase, seed, and action schedule
- Output head precision and effective bits; baseline and probe compute
- Fixed monitored task distribution and strictly matched checkpoint family

## Primary outcomes

(1) Exact control: compare minimum cross-family Hamming distance
for raw and audit codebooks against OIC-T06 and T08. Test that for
even n the fixed full-probe then passive-tail schedule achieves
d_raw(T)=T-n+1. Verify the odd-n terminal parity obstruction.

(2) Learned model: at each T report empirical cross-condition
raw-output distance distribution, held-out balanced identification
error, Wilson score confidence intervals, and calibration of
decision thresholds. Report mean and worst-case over all registered
initial hidden phases separately. A sample mean cannot stand in
for a uniform worst-case certificate.

(3) Probe efficiency: total active probes q and passive readouts T,
including the cost of activating the monitor, computing a readout,
GPU steps, CPU transfers, and memory residency. A visible
state difference retained during a passive tail is a *latch* only
if it persists after intervention stops and across matched seeds.

## Required controls

1. Odd-n cycles with all the same interface and compute budget.
2. Hidden-phase order scrambled or resampled after the probe burst.
3. Intervention/readout labels randomly permuted, including a
   no-phase baseline.
4. Event-only audit monitor with matched input corruption and
   explicitly recorded fault units; never identify one raw-bit fault
   with one event-bit fault.
5. Constant-output and two-state minimal realization controls.
6. Random and periodic passive/active schedules with the same q/T.
7. Check recovery under independent measurement errors versus
   adversarially chosen worst-case error placement.

## Acceptance and falsification

For Tier E, acceptance requires all exact Python assertions and
pinned receipts to match. For Tier L, register at least 5 seeds
per n; hold out at least two hidden-phase initializations when
there are enough states and use fresh randomized test data.

A positive *neural latch* finding requires:
- an independent probe-free demonstration of baseline trace equivalence,
- reliable separation after a fixed, frozen active burst,
- persistence for at least twice the active burst length,
- higher held-out identification accuracy than a matched random
  fixed-action schedule, and
- clear resource accounting for every monitor step.

Failure to retain separation, phase-label leakage, a changed baseline
transcript, or a result dependent on a privileged hidden-state oracle
falsifies the corresponding transfer claim. Report failure and
uncertainty rather than re-defining the sensor after seeing the result.

## Reproduction starting point

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python research/observational_inwardness_v1/run.py --check
python research/observational_inwardness_v1/robust_run.py --check
python research/observational_inwardness_v1/raw_run.py --check
~~~

The finite model is a deterministic synthetic benchmark, with exact
written proofs. Tier L training and Tier P pretrained evaluations are
future experiments and require separate experiment logs, commit pins,
model provenance, and compute receipts.

Related: [RC-01](../recurrent_circuits_v1/THEORY.md),
[LSV-10](../levin_sieve_v1/pretrained_trm/GOAL_HORIZON.md),
[6GW-T01](../../6GW/THEORY.md).
