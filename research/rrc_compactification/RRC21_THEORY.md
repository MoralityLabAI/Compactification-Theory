# RRC-21: Selective source-shift routing with exact finite verification

## Formal task and information boundary

A public topology \(\tau\) and two demonstrations \((F_j,G_j)\), \(j=1,2\),
are supplied through degree \(T=12\). Each \(F_j\) lies in
\(1+q\mathbf Z[[q]]\). The previous RRC-18 normalization groups the
\(4^3=64\) graph spellings into 25, 24, 33 or 24 exactly equivalent
*degree-12 polynomial* behaviors, depending on \(\tau\). For a representative
program \(p\), its finite certificate is

\[
 C_T(p,X)=\bigwedge_{j=1}^2 [p(F_j)\equiv G_j\pmod{q^{T+1}}].
\]

The verifier computes integer coefficients exactly, on **both** public
demonstrations. The model ranks candidates using only those observations;
future query coefficients (degrees 13–32), source-family names and the
original graph are excluded. A future success is graded independently.

## Known-only nonconformity and finite-sample result

Let \(\phi(X)\in\mathbf R^{10}\) be a fixed public feature function of the two
input series, using sparsity, sign and coefficient magnitudes. A reference
set \(R=\{r_i\}_{i=1}^{420}\) supplies empirical feature means \(\mu_j\) and
floored diagonal scales \(s_j\ge .05\). The *nonconformity* is the nearest
reference distance

\[
 D(X)=\min_{r\in R}\sum_{j=1}^{10}
 \left(\frac{\phi_j(X)-\mu_j}{s_j}-
       \frac{\phi_j(r)-\mu_j}{s_j}\right)^2.
\]

From 180 *separate* known-family calibration examples, compute
\(d_1,\ldots,d_n\). For any new case use

\[
 p_{\rm OOD}(X)=\frac{1+\#\{i:d_i\ge D(X)\}}{n+1}.
\]

**Conformal rank lemma.** Conditional on the independently chosen reference
set, if the \(n\) calibration nonconformity values and the next test
nonconformity value are exchangeable, then
\(\Pr(p_{\rm OOD}\le\alpha)\le\alpha\) for every \(\alpha\in[0,1]\).
For distinct values, the new value's rank is uniform; the displayed
non-randomized tie convention makes the p-value conservative. This lemma
guarantees false-flag control for *exchangeable known* inputs. It says
nothing about detection power on a new source generator, the correctness
of a neural program, or source-conditional guarantees. This experiment's
fixed, equally-sized family strata are a designed evaluation, so the
actual cohort-level false-positive and true-positive rates are reported
empirically rather than asserted from the lemma.

An additional *descriptive* confidence threshold is the known-only
80th percentile of the normalized entropy of the uniform mixture of
three frozen neural class distributions and the certified class-size
prior. The calibration found \(H_{0.8}=0.9094310009903586\). Entropy is
computed only when the OOD detector lets the model proceed. The two
thresholds are combined by logical OR: if \(p_{\rm OOD}\le0.1\), defer;
otherwise if entropy exceeds its threshold, defer. No conditional
coverage/risk claim follows for the conjunction or the selected neural
cases without further assumptions.

## Computational deferral and exactness

An RRC-21 policy first pays for a comparison with 420 reference vectors.
If the OOD gate flags the case, it skips neural inference. Otherwise
it pays for two GRU and one feedforward forward passes and computes the
uniform mixture ranking and its entropy. Candidate execution occurs
*only after routing*, using the preexisting RRC-18 exact interpreter.

The **strict-12** policy selects either the neural top-12 ranking or
the certified class-size top-12 ranking, both with at most twelve
full, two-demonstration checks. The **selective-full** policy makes
the same routing decision but grants deferred cases the *entire*
certified class catalogue, up to 33 checks. These are different
resource contracts and must remain separate in every comparison.
The full-class solver is complete for the chosen finite DSL: it
evaluates at least one representative of every exactly certified
polynomial class. Because all candidates are finite, its returned
program satisfies \(C_{12}\) whenever it accepts and its abstention is
complete **within the declared 64-program grammar**.

A larger verification budget can strictly expand the set of programs
that can be recognized. Therefore improved accepted counts under
selective-full cannot by itself establish a better learned algorithm
than exact synthesis. In this run complete symbolic search on every
positive case attains the full 480/480 budget-unconstrained ceiling;
the selective policy trades some of those successes for fewer exact
checks on average while paying for calibration, model and routing costs.

## Future-collision bound

Take two records with exactly the same demonstrations, public topology,
query and calibrated router inputs. Change one record's unseen query
output by \(q^{15}\). Every deterministic selection policy executes the
same graph checks and emits the same program for both. At most one
output can match both distinct continuations, so at least one of each
*accepted* pair is a future error. A finite prefix certificate never
proves a continuation to degree 32 or an infinite formal-series identity.

## Accounting

The calibration stores 420 ten-coordinate reference vectors, 180 scalar
nonconformity values, 180 entropy values and public normalization
statistics. Each test OOD distance charges a nominal lower-bound
\(3\cdot420\cdot10=12,600\) floating scalar operations, excluding
feature construction, reference scanning overhead and sorting. The
uniform three-neural mixture charges an estimated \(112,552\) dense
floating-point multiply–accumulates per model-inferred case. The
per-candidate exact coefficient-work proxy follows the topology-specific
RRC-16 program interpreter; an exact graph query executes two public
demonstrations. The offline class compiler performs 256 symbolic graph
normalizations and is shared with previous experiments.

The three frozen experts together store 18,715 FP32 parameters. Their
historical RRC-19 training used a separate budget; RRC-21 applies no
weight updates. Interpreter source, class tables, reference features,
calibration data, model weights, sorting, big-integer costs, and test
generation all require storage or computation. Floating-point MACs,
scalar feature arithmetic, integer coefficient-loop proxies and actual
latency cannot be summed into a single hardware-neutral cost.
