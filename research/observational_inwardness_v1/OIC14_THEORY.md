# OIC-T14–T17: learnable circle dynamics and discrete-group retraction

**Date:** 2026-11-10. **Status:** elementary exact-real derivations, conditional numerical certificates and executed CPU synthetic-recurrence pilot. The claims are about a stated model family, with known initial phase encoding and a known observable XOR rule. No Lean kernel compilation or pretrained-model experiments are claimed.

## Model and observation

For `n >= 4`, the target hidden phase is `z_t in Z_n`. Fixed targets have `z_(t+1) = z_t`; rotating targets have `z_(t+1) = z_t+1 mod n`. The visible transition is

`x_(t+1) = x_t XOR 1 XOR (u_t AND [z_t=0])`.

The learnable orthogonal model starts with `h_0=(cos(2πz_0/n),sin(2πz_0/n))` and updates `h_(t+1)=R(omega) h_t`. Its learned event detector is `g(h)=1[w*h_x+b >= 0]` at evaluation. During training the detector is replaced by `sigmoid(w*h_x+b)`, with teacher-forced visible state inputs. Only the visible state transition is supervised; the model sees the encoded initial hidden phase. The group representation and binary XOR output schema are architectural assumptions. The rotation angle and detector parameters are learned.

The comparator uses the same input and detector, but replaces `R(omega)` by an unconstrained learned real `2x2` matrix `A` (6 trainable scalars vs. 3). The group-likelihood grid additionally searches `4n` candidate angles on the **first training minibatch** before running the same 350 Adam steps. That search adds inference work beyond the other approaches.

## OIC-T14. Exact equivariance of every learned orthogonal update

Let `Q_k=R(2πk/n)` for `k in Z_n`. Every planar rotation commutes with every other planar rotation, so

`R(omega) Q_k h = Q_k R(omega) h`

for all real `omega`, `h in R²`, and `k in Z_n`. In addition `R(omega)^T R(omega)=I`, which preserves `||h||_2` exactly over the real numbers for any number of steps. These are structural identities independent of the optimizer's output. The phase-zero event detector is *anchored* and need not be invariant under `Q_k`.

The unrestricted matrix generally has neither property. In that comparator we record both `||A^T A - I||_F` and `||A Q_1-Q_1 A||_F` as diagnostics.

## OIC-T15. Conditional group-retraction theorem

Let `omega_* = 2πs/n` be the true target group generator, where `s in {0,1}` for this benchmark. Given the *learned* `omega_hat`, define

`q(omega_hat) = round(n*omega_hat/(2π)) mod n`.

For a unique nearest lattice angle, if the wrapped angular distance obeys

`dist_circle(omega_hat,omega_*) < π/n`,

then `q(omega_hat)=s`. If moreover `sign(w cos(2πz/n)+b)` gives the correct event decision `[z=0]` for **each** `z in Z_n`, then the projected finite-state model updates by `z_(t+1)=z_t+q(omega_hat) mod n` and reproduces every clean observable history of the exact target for every initial phase and every action word, regardless of length.

**Proof.** The nearest-lattice condition puts `omega_hat` in the Voronoi cell of the correct group generator, so the rounded generator is `s`. At time zero the projected phase equals the true phase. Inductively both phases advance by `s`, hence agree at every step. The anchored event detector is correct at each such lattice phase, so the XOR readouts agree. QED.

This is a *conditional retraction theorem*, not a claim that training always lands in the correct cell. The true generator is used only for post hoc evaluation of the condition; the projection algorithm itself uses `omega_hat` and the known integer `n`.

In the OIC-14 archived pilot, the conditions hold for all 60 orthogonal-grid fitted models and 55 of 60 locally optimized orthogonal models. The resulting infinite-horizon claim is for the **projected** finite-state machine, using exact integer phase updates and a fixed finite readout table; it is not a bit-exact guarantee about floating-point iterative angle multiplication.

## OIC-T16. Angular-drift obstruction before projection

Compare the learned orthogonal recurrence with the target phase rotation. Let `delta=omega_hat-omega_*`. Their unit-circle states after `t` transitions satisfy

`||R(omega_hat)^t h_0 - R(omega_*)^t h_0||_2 = 2|sin(t delta/2)| <= t|delta|`.

The scalar detector logit `w*h_x+b` is `|w|`-Lipschitz in `h`. Suppose the detector correctly classifies every true lattice phase and has a minimum positive absolute logit margin

`gamma = min_(z in Z_n) |w cos(2πz/n)+b| > 0`.

Then all event decisions are preserved at all times `t` such that

`t |w| |delta| < gamma`.

The bound is sufficient and may be conservative. At long horizons, even a small learned angular error can cross a decision boundary, disrupting raw visible output when the probe is active. This explains why a model can attain exact registered `n-probe, n-passive` latch trajectories while failing a longer continuously probed sequence. Both the model's angular parameters and the probe schedule matter.

For an unrestricted linear update `A` and target orthogonal update `R_*`, telescoping gives

`||A^t - R_*^t|| <= ||A-R_*|| sum_(j=0)^(t-1) ||A||^j`.

The bound can grow exponentially when `||A||>1`. This is an ordinary matrix-power perturbation inequality and requires no new systems theory.

## OIC-T17. Parameter-precision margins for the projected machine

Write `omega_q=2πq/n` for the nearest group generator and define the angular *cell margin*

`m_angle = π/n - dist_circle(omega_hat,omega_q)`.

When `m_angle>0`, any perturbation to the stored angle of magnitude **strictly less** than `m_angle` leaves the selected group generator unchanged. For the anchored readout define

`m_gate = min_(z in Z_n)|w cos(2πz/n)+b|`.

Since `|cos(theta)|<=1`, simultaneous parameter errors `delta_w`, `delta_b` with `|delta_w|+|delta_b| < m_gate` preserve every lattice-phase event classification. Combined with the OIC-T15 conditions, these inequalities certify output equivalence after projection, within the real arithmetic model.

If each real parameter is rounded to an absolute grid of spacing `2^-f`, then its maximum error is `2^(-f-1)` (assuming ordinary nearest rounding and no overflow). A sufficient number of **fractional bits** satisfies

`2^(-f-1) < m_angle` and `2^(-f) < m_gate`.

Integer-range bits and the cost of storing `n`, the phase generator, the readout schema and the decoder must be accounted separately. These bounds concern fixed-point precision of a small synthetic recurrence, not general neural compression or physical ontology.

## Interpretation boundary

The mathematically substantive object here is the relation among a continuous recurrent update, a known finite cyclic action, observed events and a discrete projection. Classical symmetry, coding and system-identification mathematics supply the ingredients. The experiment asks how far a carefully structured learner can recover hidden periodic dynamics from visible supervision and how much brittle drift remains without algebraic projection.

References: OIC-01–13 under this directory; OIC-11 preregistered neural-monitor protocol; and existing `research/levin_sieve_v1/pretrained_trm/GOAL_HORIZON.md` and `research/recurrent_circuits_v1/THEORY.md` for observability and resource accounting.
