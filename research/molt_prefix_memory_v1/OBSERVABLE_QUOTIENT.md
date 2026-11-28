# Local observable quotients and memory allocation by visit

Development mathematical note for MOLT-06. Companion:
[LSA calculus](https://github.com/MoralityLabAI/LoopedTransformers/blob/c25fe023b40eb4a802f4bb733f777d6d68067c88/docs/lsa_typed_architecture_calculus_v0.md),
[SPC-06C](https://github.com/MoralityLabAI/LoopedTransformers/blob/c25fe023b40eb4a802f4bb733f777d6d68067c88/experiments/spectral_state_cost_v0/spc06_future_observability.py),
and [the exact prefix fixture](README.md).
No Lean proof or new native-checkpoint result is claimed.

## 1. Fixed cut, support and observer

Freeze the input, epoch, physical parameter objects, reached state, future schedule
and readout family. In the architecture record A=(S,Phi,w,rho,g,sigma,pi), pi
is carry and kappa_g is gradient alignment. Let q_j be the readout at visit j.
A differentiable suffix induces the stacked observer F: X_r -> R^m.
For an orthonormal support injection E: R^d -> X_r, set

    O = DF(s_r) E,   G = O^T O,   d_obs = rank O.

The quotient R^d / ker O is a first-order observable tangent space. The
finite-horizon nonlinear fibers are level sets of F, generally curved.
The constant-rank theorem supplies a local fiber chart where the rank
is constant; exact recurrent lumpability on an invariant set needs the
existing stronger condition that the transition preserve readout fibers.
An observer can have a small derivative rank at one point and distinguish
directions at second order (e.g. F(x,y)=x+y^2 at (0,0)).

## 2. Nested horizons and spectra

For a fixed cut and fixed support, append additional observer rows J:

    O_new = [O; J],  G_new = G + J^T J.

Hence ker O_new = ker O intersect ker J. The ordered eigenvalues of G_new
are at least those of G by the min-max principle. Exact rank is nondecreasing.
Numerical rank at a fixed absolute singular-value threshold is also
nondecreasing; a relative threshold recomputed from each largest singular
value need not share this property. Save the threshold and full spectrum.

These comparisons are for the same cut and fixed reached trajectory.
Across cuts, changing the intervention chart, or changing the execution
word, the domains and transported observers change. Do not infer monotone
rank in that different comparison.

## 3. Optimal local rank-k memory sketch

Write an SVD O=U diag(sigma) V^T with singular values in decreasing order,
zero-padded to the support dimension. Let P_k project onto the first k
right singular vectors. Among rank-k linear reconstructions R, the
smallest possible spectral norm of O(I-R) is sigma_(k+1). The upper bound
is attained by P_k. For the lower bound, the span of the first k+1 right
singular vectors intersects ker R nontrivially; a unit vector v there
satisfies ||O(I-R)v||=||Ov|| >= sigma_(k+1).
The same truncation minimizes squared Frobenius error by the usual
Eckart-Young theorem.

For perturbations ||d||<=a, the local first-order error from retaining P_k d
is at most a sigma_(k+1). If the norm of the restricted second derivative
of F(s_r+E v) is bounded by M throughout the support ball of radius a,
Taylor's theorem gives

    ||F(s_r+E d) - F(s_r+E P_k d)||
       <= sigma_(k+1)||d-P_k d|| + M a^2.

Indeed subtract the two linearizations at s_r; each remainder is bounded
by M a^2/2. This bound is conservative, valid only under the stated
uniform smoothness condition. For scalar utility scores whose uniform
errors are at most epsilon, a strict margin >2 epsilon across the top-k
boundary certifies unchanged selection; exact environment verification
remains mandatory for hard deductions.

The sketch compresses an *intervention* under this observer. It does not
compress arbitrary model states, attest a memory-size reduction, or prove
that approximate reconstruction preserves an entire recurrent computation.

## 4. Match feasibility and higher-order sensitivity

Given target d and recipient z in the same orthonormal chart, a strong
sham matches signed O d, ||d|| and ||z+d||. With C=[O;z^T], the solutions
are the affine sphere p+{v in ker C: ||v||=a}, where p is the orthogonal
projection of d onto row C and a=||d-p||. If a=0 the solution is unique.
If a>0 reflection across p is a distinct solution, including when the
nullspace is one dimensional. If the nullspace has at least two
dimensions one can rotate the residual orthogonally. Additional physical
constraints can eliminate all distinct feasible solutions; they must
be audited rather than relaxed after the result.

For a twice differentiable F,

    F(z+t d)-F(z+t s)
      = (t^2/2)(D^2F(z)[d,d]-D^2F(z)[s,s]) + o(t^2)

when O d=O s. This does not establish finite-dose equality.
Use multiple central-difference step sizes, and measure the actual
nonlinear remainder as well as verified task outcomes.

## 5. MMS/typed lattice handoff

A memory map may record an epoch key, checkpoint, cut address, support,
observer, singular vectors/values, retained rank, reconstruction cost,
and certified local validity radius/error allowance. The spectrum supplies
a candidate memory priority. It is invalidated when its binding changes,
unless a separately tested approximation contract covers the change.
For an orthogonal component projector P_C, observable energy is
tr(P_C G P_C); disjoint latent subspaces can still interfere after O,
so include the cross terms when measuring combined effects. General BPD
factors are not necessarily orthogonal projectors.

Rank and byte budgets are different: a basis, singular values, retained
coordinates, epoch keys, and construction scratch all cost memory. Actual
MMS comparison requires fixed CPU quota, measured process-tree RSS and
verified-success/work receipts. A hard LDT refinement must still pass
the independent membrane; local observability is a soft scheduling signal.

Executable development instrument:
[LoopedTransformers MOLT-06](https://github.com/MoralityLabAI/LoopedTransformers/pull/403).
The note introduces no new confirmed architecture advantage.
