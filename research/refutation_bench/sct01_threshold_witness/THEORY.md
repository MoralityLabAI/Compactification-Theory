# SCT-01: SAT expectation covers, hitting times, and proof-carrying memory

Date: 2026-11-14. Status: new finite mathematical deductions with exhaustive
rational and combinatorial reference checks. Original OpenAI source theorems
remain research claims; this campaign neither proves their general results
nor extrapolates the finite instance into a random-SAT asymptotic theorem.

## Source pair and precise mathematical scope

**OpenAI Math #175: Talagrand's expectation thresholds and discrete
convexity**, September 23, 2026.
https://github.com/openai/math/blob/main/lean/docs/175.md

The selected formalized theorem compares the integral and fractional
expectation thresholds of any nonempty proper increasing family of
subsets of a finite nonempty ground set, with cover budget 1/2:
q_f(F) ≤ 25·512⁴ q(F). Fractional covers may weight every subset,
including ∅. Other selected statements concern discrete convexity.
The finite theorem below calculates *both* cover optima exactly
for a special monotone UNSAT family. It does not apply or reprove
the general comparison bound.

**OpenAI Math #235: Limiting random SAT thresholds, sharp variance and
computability**, September–October 2026.
https://github.com/openai/math/blob/main/lean/docs/235.md

The selected source formalization treats proper k-clauses on k distinct
variables, independently sampled *with replacement*, establishing
limiting satisfiability thresholds for fixed k≥3 and statements on
the variance of first UNSAT hitting time. For k=3 the September
variance paper gives linear lower and O(n log n) upper bounds,
with a separate later companion claiming a sharper result. The
source also reports computability of the limiting 3-SAT threshold.

We work with the exact full 3-SAT clause universe at **n=3**
as a calibration, and with a restricted six-variable, two-block
clause distribution as the main experiment. The latter is *not*
the full uniform random 3-SAT law on six variables. No
asymptotic source theorem is imported to assign its exact
finite probability or variance.

## Construction: disjoint complete 3-SAT cores

Let r≥1. Partition n=3r variables into r disjoint three-variable
blocks. Within each block include the eight proper signed
3-clauses, indexed by the unique local truth assignment they
exclude. The total clause universe has M=8r distinct types.

For any subset A of this universe, the conjunction of the
included clauses is satisfiable precisely when each block has
an unexcluded local assignment. Consequently

\[
\boxed{
\mathrm{UNSAT}(A)
\iff\exists g\in\{1,\ldots,r\}:\ C_g\subseteq A,
}\tag{SCT-1}
\]

where C_g is the complete eight-clause set on gadget g.

**Proof.** Each clause indexed by a local assignment a is false
precisely on a. A proper subset of a gadget's eight clauses leaves
some local assignment available. Since the variable blocks are
disjoint, choose one available assignment per incomplete block
and concatenate them. A complete gadget forbids every local
assignment, rendering the whole conjunction unsatisfiable. QED.

Thus the *only* inclusion-minimal UNSAT subformulas are the
r disjoint cores C_g, all of cardinality eight.

The reference independently enumerates all 2^16=65,536
subformulas for r=2 and uses a 64-assignment bitset SAT oracle,
without relying on the formula SCT-1 to evaluate satisfiability.
It finds 511 UNSAT and 65,025 SAT subformulas, exactly two
minimal UNSAT cores.

## SCT-2: exact fractional and integral expectation covers

Let X be the M clause types and let F be the increasing family
of UNSAT clause subsets. A (containment) cover assigns
nonnegative weights w_S to subsets S⊆X such that
Σ_{S⊆A}w_S≥1 for every A∈F. Its p-cost is
Σ_S w_S p^{|S|}. An **integral cover** restricts the weights
to zero or one, while a fractional cover permits nonnegative
real weights. The empty set is allowed and has cost one.

For the r disjoint inclusion-minimal cores of size eight,

\[
\boxed{
C_{\mathrm{fractional}}(p)
=C_{\mathrm{integral}}(p)
=\min\{1,rp^8\},\quad0\le p\le1.
}\tag{SCT-2a}
\]

**Proof of lower bound.** Test the cover on each minimal member
C_g separately. Put z=w_∅. Every nonempty S⊆C_g
has cost p^{|S|}≥p^8 for 0≤p≤1. Since the cores
are pairwise disjoint, no nonempty S is contained in two
distinct C_g. For 0≤z≤1, the total p-cost is therefore
at least z+rp^8(1-z), which is at least min(1,rp^8).
For z>1 the empty set itself costs more than one.
All other terms are nonnegative. The same lower bound
applies to integral covers.

**Construction.** Choose the single empty witness
(cost one) or all r complete cores (cost rp^8),
whichever is cheaper. This is an integral cover, so
both optima attain the same bound. QED.

The 1/2-budget expectation thresholds therefore coincide:

\[
\boxed{
q_f(F)=q(F)=(2r)^{-1/8}.
}\tag{SCT-2b}
\]

The finite family has *zero integrality gap*.
This special-case equality is consistent with #175
but does not imply its universal comparison theorem.

## SCT-3: probability threshold differs even with zero cover gap

In the Bernoulli p-model each of the 8r clause types is
included independently with probability p. Events that
each disjoint core appears are independent. Therefore

\[
\boxed{
\Pr_p(\mathrm{UNSAT})=1-(1-p^8)^r.
}\tag{SCT-3a}
\]

Its 50% UNSAT probability threshold is

\[
\boxed{
p_{1/2}=(1-2^{-1/r})^{1/8}.
}\tag{SCT-3b}
\]

For r=2, this becomes 2p^8-p^16; the expectation-cover
threshold is 4^{-1/8}≈0.840896, while the probability
median is ≈0.857706. The two definitions answer
different questions, even though the fractional and
integral optimum costs agree.

The source independently counts every UNSAT subset by
cardinality. At M=16, the coefficient is

\[
\boxed{
W_j=2\binom8{j-8}-\mathbf1_{\{j=16\}}
\quad (8\le j\le16),}
\tag{SCT-3c}
\]

and zero below j=8. Hence
Σ_j W_j p^j(1-p)^{16-j}=2p^8-p^16
by an independent Bernstein-basis calculation.

## SCT-4: exact stopping-time distribution with replacement

Now sample one of the 16 clause types uniformly at every
time step, independently *with replacement*. Let H be the
first draw index at which either eight-clause gadget
has been collected completely.

For a specified set of a clause types, let F_a(t) be
the probability all of its types appeared during t
draws from the M=16 universe. Inclusion-exclusion gives

\[
F_a(t)=\sum_{j=0}^{a}(-1)^j
\binom aj\left(\frac{16-j}{16}\right)^t.
\]

The two gadget-completion events have the same
probability F_8(t); their intersection is the event
all 16 types have appeared, of probability F_16(t).
Thus

\[
\boxed{\Pr(H\le t)=2F_8(t)-F_{16}(t).}
\tag{SCT-4a}
\]

The finite exact median is 31 clause draws.

**Independent oracle.** Conditional on the final set of
distinct observed clause types having cardinality j,
the number of draw sequences with exactly that
set is j! S(t,j), where S is a Stirling number of
the second kind. Summing over all UNSAT sets yields

\[
\boxed{
\Pr(H\le t)
=\frac{1}{16^t}\sum_{j=0}^{16}W_j j!S(t,j).
}\tag{SCT-4b}
\]

The reference calculates both rational expressions
independently for multiple t, including 31 and 32,
and requires exact equality.

**Moments.** Write \(a_j=(16-j)/16\) and
\(c_j=(-1)^j[\binom{16}{j}-2\binom8j]\),
with the second binomial interpreted as zero for j>8.
Then

\[
\Pr(H>t)=\sum_{j=1}^{16}c_j a_j^t.
\]

Using Σ_{t≥0}a^t=1/(1-a) and
Σ_{t≥0}(2t+1)a^t=(1+a)/(1-a)^2,

\[
\begin{aligned}
E H &=16\sum_{j=1}^{16}\frac{c_j}{j}
=\boxed{\frac{1481069}{45045}}\approx32.879765,\\
E H^2&=16\sum_{j=1}^{16}
\frac{(32-j)c_j}{j^2},\\
\operatorname{Var}(H)
&=\boxed{\frac{80211997256}{676350675}}
\approx118.595279.
\end{aligned}\tag{SCT-4c}
\]

A *third* implementation computes the mean and
second moment by backward Markov dynamic programming
over all 65,536 clause-presence states and checks
the exact formulas within floating-point tolerance.

At n=3 (only eight total proper 3-clauses), the
full random 3-SAT model's first UNSAT time is exactly
the ordinary eight-coupon-collector time. Its
moments are 761/35 and 838034/11025.
This special n=3 fact does not establish the
large-n variance behavior studied in #235.

## SCT-5: exact deterministic online state complexity

A monitor reads a stream of clause types from the
fixed universe of r disjoint gadgets of k distinct
types, with arbitrary repetitions, and must decide
SAT/UNSAT correctly after **every possible stream**.

Presence sets that already contain a complete
gadget are all equivalent for this binary decision:
future clauses preserve UNSAT. A *losing* presence set
is one with every gadget incomplete; there are

\[
(2^k-1)^r
\]

such sets. Any two distinct losing sets A,B can be
distinguished by a continuation: choose a clause
x∈A\B (swapping A and B if needed), let C_g be
its gadget's complete clause set, and append
all elements of C_g\A. The resulting A-stream
completes C_g. The B-stream still lacks x
and cannot complete any other gadget because
the suffix contains only clauses of C_g.
Thus A and B must occupy different deterministic
monitor states.

The winning state is also distinct from every
losing state because the current output differs.
The Myhill–Nerode argument proves the exact minimum:

\[
\boxed{
N_{\mathrm{decision}}=(2^k-1)^r+1,\qquad
B_{\mathrm{decision}}=
\lceil\log_2 N_{\mathrm{decision}}\rceil.
}\tag{SCT-5}
\]

**Matching implementation.** Encode each incomplete
gadget's clause-presence mask, an integer from
0 through 2^k-2, as one base-(2^k-1) digit.
When any gadget completes, enter one absorbing
UNSAT state. The automaton occupies exactly
the lower-bound number of states and updates
each newly arriving clause by one digit.
The reference checks all 2^16 clause masks
and every one of their 16 possible next clauses.

For r=2,k=8:

\[
\boxed{N=255^2+1=65026,\quad B=16\text{ bits}.}
\]

The one-bit final decision thus has a substantially
larger worst-case *live* memory requirement.
The result concerns deterministic exact stream
recognition with unbounded input length. It does
not constrain approximate, probabilistic or
distribution-restricted monitors.

## SCT-6: exact proof-carrying monitor and witness cost

Strengthen the required output: on SAT inputs the
monitor must say SAT; on UNSAT inputs it must
output a **valid index** of one of the completed
gadgets, which a trusted verifier can expand
into its eight original clauses.

Each pair of losing supports remains distinguishable
by SCT-5. Further, a stream containing precisely
one complete gadget C_g has only one valid output
index g. These r distinct positive inputs cannot
share an accepting state. Accepting and losing
states are also distinct. Therefore

\[
\boxed{
N_{\mathrm{certifying}}=(2^k-1)^r+r.
}\tag{SCT-6a}
\]

This bound is attained: use the same incomplete
base-(2^k-1) digits, but when the first gadget
completes enter one of r absorbing states labeled
by that gadget. Further clauses cannot invalidate
that witness, even if other gadgets later complete.

For r=2,k=8, the minimum is 65027 states, still
16 fixed-length live bits. A positive proof needs
just one gadget identifier bit, **if the public
static decoder already contains the gadget's eight
clauses**. Its codebook, source-program description,
input stream and verification access are additional
resources.

On a restricted source prior uniformly supported
on exactly-one-full-gadget streams, the success of
any terminal B-bit witness decoder with independent
randomness is at most

\[
\boxed{
P_{\mathrm{correct\ index}}\le
\min(1,2^B/r).
}\tag{SCT-6b}
\]

Condition on all independent coins. A decoder
with at most 2^B outputs can name at most that
many different correct gadget indices, and
each source index has prior 1/r. Averaging
preserves the bound. Exact identification
therefore needs at least ceil(log2 r) bits
under that promise, a different task from
online certification of all arbitrary streams.

## SCT-7: implications for RefutationBench

The test setting can give a reasoner a stream of
proper 3-clauses, repeated in arbitrary order.
The host owns source clause semantics and validates
the candidate proof. Measure separately:

- the first time the stream actually becomes UNSAT,
  comparing to the exact hitting-time law;
- correctness of the binary SAT decision on
  arbitrary prefixes, including adversarially
  repeated clauses;
- whether a claimed eight-clause core actually
  appeared in the input;
- live monitor memory, witness identifier bits,
  static core decoder, source input storage
  and verification computation.

A naive clause-arrival counter can report UNSAT
after eight draws of the *same* clause. It violates
the deterministic exact recognition contract.
The certified base-digit monitor tracks distinct
clause types and never invents a contradiction.

For a full random k-SAT campaign, remove the
disjoint-gadget promise and use the original
proper signed clause distribution on all variable
triples. The source #235 limiting threshold
and variance results require that setting and
large-n analysis. General UNSAT proof discovery
may have much larger minimal cores and different
complexities. The present experiment provides
a complete finite control for those later
benchmarks, not a result on their scaling.

The source #175 comparison theorem is an external
general claim; our exact disjoint-core cover
shows equality of its two cover notions on
one controlled increasing family. This is an
illustration of its objects, not a restatement
of the universal inequality.

No trained Lean reasoner, native neural checkpoint
or production theorem prover was evaluated in
this phase. The Lean formal companion, if present,
certifies finite gadget and automaton arithmetic,
while the full probability derivations and
general minimal-automaton proof retain their
separate mathematical scope.
