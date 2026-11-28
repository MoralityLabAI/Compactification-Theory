# MTB-01: Markov type, Brenier stability and semantic drift budgets

Registration: November 14, 2026. This protocol precedes executable
finite experiments and training. This campaign runs no trained checkpoint.

## Source pair

- OpenAI Math #327, *Markov type characterizes superreflexivity*:
  nontrivial Markov type p>1 of a real Banach space is equivalent to
  superreflexivity, with the existence of a corresponding equivalent
  uniformly convex norm. This is a source research theorem.
- OpenAI Math #374, *Sharp one-third stability of Brenier maps*:
  for uniform source on a fixed compact convex body in R^d, d>=2,
  and targets supported in a fixed compact set, L2 displacement
  between quadratic Brenier maps is <= C W2(targets)^(1/3).
  The exponent is optimal across those target classes.

The following synthesis has its own proof, conditional on those
source statements, and the finite examples use direct algebra.

## Target theorem

Let (X_t) be stationary reversible with uniform law on compact convex
Omega ⊂ R^d, d>=2. Let T_mu,T_nu be Brenier maps from the uniform
source into target measures mu,nu supported in one fixed compact K.
Let A:R^d→Y be L-Lipschitz, and let the Banach space Y have Markov
type p in (1,2] with constant M_p(Y). Then

  [ E || A(T_nu(X_t))-A(T_mu(X_0)) ||_Y^p ]^(1/p)
      <= L*C_K*W2(mu,nu)^(1/3)
       + M_p(Y)*t^(1/p)*E_1^(1/p),

where E_1 = E||A(T_mu(X_1))-A(T_mu(X_0))||_Y^p.
This follows by Minkowski, source #374 and the Markov type
inequality. For Hilbert Y, p=2 and M_2=1, with an independent
elementary spectral proof. The bounds compare full transported
representation values, and they do not establish optimal readout
or hidden reasoning in real language models.

A decision with an initial local-constancy radius m for all but
epsilon of the source states has flip probability bounded by
epsilon+(D_t/m)^p (capped at one) via Markov inequality.
The margin and feature extractor are separate charged resources.

## Frozen exact affine Brenier / refresh control

Source domain Omega=[0,1]^2 with continuous uniform law.
Target mu is uniform on the rectangle [0,a]×[0,b] and target nu
uniform on [0,c]×[0,d], a,b,c,d>0 rational. Brenier maps are
T_mu(u,v)=(a*u,b*v), T_nu(u,v)=(c*u,d*v).

Each step independently refreshes each source coordinate from
Uniform[0,1] with probability s and otherwise retains its previous
value. The stationary reversible chain has uniform source.
After t steps, the same-coordinate persistence is q=(1-s)^t.

The exact analytic cross-transport squared drift is

  E||T_nu(X_t)-T_mu(X_0)||²
    = ((c-a)²+(d-b)²)/3 + (a*c+b*d)*(1-q)/6.

Its W2(mu,nu)² equals the first term. Same-target drift is
(a²+b²)*(1-q)/6. One-step Markov budget is
(a²+b²)*s/6, and the t-step type-2 inequality follows from
1-(1-s)^t<=t*s.

Freeze parameter cells:
  (a,b,c,d,s)=(1,1,101/100,1,1/100)
  (a,b,c,d,s)=(1,1,2,1,1/4)
  (a,b,c,d,s)=(3/2,2,2,5/2,1/4)
and time horizons t=0,1,2,4,8,16.
All probabilities and moments are exact rational fractions.

For a first-coordinate decision threshold theta=1/2, compute
exact probability of a decision changing under the noisy refresh
and rescaled Brenier map; report a local margin m=1/5 event and
the Markov bound. Test t=0,1,2,4,8,16. Distinguish the exact
affine-family W2 stability exponent 1 from #374's worst-case
universal exponent 1/3.

## Frozen hypercube ℓ1 counterexample

On the n-bit cube {0,1}^n with uniform stationary reversible
simple random walk, flip one uniformly chosen coordinate per step.
Embed the cube in ℓ1^n with the Hamming distance.

At t steps let a=(1-2/n)^t and b=(1-4/n)^t. Then
  p_t=(1-a)/2,
  E H_t = n*p_t,
  E H_t² = n*p_t + n(n-1)*(1-2a+b)/4.

At t=n, the exact Markov-type-2 ratio is
  R_n=E H_n²/(n*E H_1²)=E H_n²/n.

By Jensen, for any p>1, an ℓ1^n Markov-type-p constant
is at least E H_n/n^(1/p). Since
E H_n>=((1-e^-2)/2)n, the required dimension-dependent
constant grows like n^(1-1/p). In particular the family
of ℓ1^n norms cannot have a dimension-uniform nontrivial
Markov type, consonant with source #327.

Register n={4,8,16,32,64,128,256}, t={1,2,4,8,n}
with duplicate times deduplicated. An independent rational
birth-death Markov chain over Hamming distance values
checks n=4,8,16 through at least t=8.
Also check dimension-independent Hilbert Markov-type-2
upper bounds using a separate coordinate embedding where
squared Euclidean distance equals Hamming distance.

## Evidence and non-claims

Exact reference uses only standard-library Python Fraction. Source theorem
statements are imported conditionally; no general Markov-type equivalence,
general sharp Brenier counterexample or Wasserstein stability result is
reproved by finite experiments. The affine Brenier controls are
closed-form cases and do not attain source #374's worst-case 1/3 exponent.

Counter/storage ledgers charge live latent state, observation time
horizon, map representation, target distribution information,
Markov-type constant (potentially dimension-dependent), readout
margin and decoder computation separately.

Any extensions to LoopedTransformers SAEs or learned reasoners must
freeze checkpoints/activation discretization, verify that observed
transitions are stationary reversible, and obtain an independently
grounded source-target transport pairing before applying the theorem.
