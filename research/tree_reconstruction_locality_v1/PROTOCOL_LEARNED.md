# KSR-01B: prospective small learned count-decoder pilot

Registration date: November 14, 2026, before running the trained
count-monitor executable. This is a newly trained **synthetic,
multiclass logistic** probe, using Python standard library scalar
arithmetic. It is not an original LoopedTransformers checkpoint
and does not approximate the infinite FIID generation theorem.

## Frozen data cells

Use the q=3 symmetric broadcast channel, branching b=2,
lambda=4/5, and a uniform root prior. Evaluate three cells:

- all 8 depth-three leaves observed;
- each of 4 depth-two leaves observed independently with r=1/2;
- depth-two sparse KSR-6 retention r=(25/32)^2=625/1024.

Original tree generation is independent conditional on each parent:
on each edge, copy parent label with probability lambda and otherwise
resample a uniformly random label. Only retained leaf counts
enter the model.

For each cell draw 4096 training and 4096 independent held-out
whole trees using fixed base seeds 940229236 and 940229237,
with each cell assigned a fixed integer seed offset.
The two fit seeds are 11 and 19. A whole tree is the independent
sampling unit; leaf bits in one tree are correlated and are not
treated as separate examples.

## Model and controls

Use a small q-class softmax linear readout on the public
feature vector consisting of q count fractions (each count divided
by total possible leaves), retained fraction, and an intercept.
The model has q*(q+2)=15 float64 learned scalar coefficients.
The mutable monitor stores the q integer counters, with the
verified bits-per-count budget in reference.bit_budget.

Fit by online SGD with six passes over the 4096 training trees,
step size 0.15, seed-controlled shuffling and zero initial weights.
Each arm makes the same number of updates.

Arms:
- supervised root labels;
- random independent uniform legal labels, supplied in a fixed
  seed-specific order and never computed from held-out roots.

Baselines:
- fixed plurality-of-leaf-counts decision with lowest-label tie break;
- exact Bayesian root MAP from the finite source oracle after
  aggregating every observed leaf pattern into count cells;
- analytic KSR-3 lower bound from the normalized score.

The exact Bayes floor is a **population lower bound**, and
exact count MAP is a population optimum. Finite held-out
sample accuracy may fluctuate above or below either,
without violating the source mathematical inequalities.

## Scoring and evidence

Report learned weight SHA256 from deterministic float64 serialization,
source and dataset SHA256, count bit budget, train/heldout accuracy,
mean heldout log loss, correct/total counts and the exact population
count MAP. Include a calibration-only choice of sign/temperature
if subsequently extended; none is tuned on this run's heldout set.

Do not compare to an unregistered larger model in the same
decision table. Because this task exposes true label-count
histograms, results concern synthetic finite-root inference,
not native neural reasoning, self-ascribed beliefs or
generalization to different channel parameters.
