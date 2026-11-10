# RRC-24: q-adic late divergence and exact factor recovery

Let \(\mathcal U=1+q\mathbf Z[[q]]\), and fix four public unit-preserving base operations
\(O\in\{I,N,R,S\}\) for identity, sign substitution, reciprocal and
\(q\mapsto q^2\) substitution. Consider the finite DSL

\[
P_{O,K,c}(F)=O(F)(1+cq^K),\qquad
K\in\{64,80,96,112,128,144,160,176\},\quad c\in\{-2,-1,1,2\}.
\]

A null delayed factor \(c=0\) adds one spelling for each base operation,
yielding \(4(8\cdot4+1)=132\) syntax programs. The observed degree is
\(T=192\), the new-query future runs through \(H=256\), and each case
supplies two public input/output demonstrations.

## The long shared prefix

For fixed \(O,F\), any two nonzero delayed factors give precisely the same
coefficients through degree 63, since their differences are divisible by
\(q^{64}\). Thus the 32 nonzero delayed variants of each base program
produce one identical length-64 coefficient prefix. A verifier that inspects
only degrees \(0\ldots63\) has **zero distinguishing information** about
\(K,c\), conditioned on the base operation and input. For degrees
\(T<K\), a delayed factor and the null factor remain indistinguishable.

## Exact valuation theorem

Let \(B(q)=O(F)(q)\) with \(B(0)=1\). For a valid nonzero delayed factor,

\[
G(q)-B(q)=cq^K B(q).
\]

The first nonzero coefficient of \(G-B\) is therefore at degree \(K\), and
its value is exactly \(c\). Recover \(K,c\) from one public demonstration,
check the proposed shift against *all* visible coefficients, and demand
that the second independent demonstration recovers the **same** pair.
For a particular base \(O\), this method identifies its only possible
nonzero delayed factor at degree \(T\), or the null factor when the difference
vanishes throughout. Its acceptance is exact to degree \(T\). The enumerator
checks that any nonnull delay lies in the declared eight-element set and any
amplitude lies in the declared four-element set. The interpreter also checks
both demonstration outputs coefficient by coefficient.

The supplied factor-recovery operation is an **algorithmic oracle** relative
to a purely learned proposal: the models of RRC-24 predict **only \(O\)**,
using coefficients through degree 16. They obtain \(K,c\) through this public
integer-arithmetic operation at test time. The factorized solver therefore
has stronger syntax knowledge and more interpreter capabilities than a
model attempting direct 132-way program identification.

## Exact and information-theoretic controls

A canonical literal 132-program search evaluates the complete DSL, with
progressive stages at degrees 16, 64, 128 and 192. This diagnostic charges
every repeated pair of demonstration executions. A full factorized search
tries the four base operators; each checked base evaluates both complete
public demonstrations and recovers its delayed factor algebraically.
The two query units differ, and all operation proxies are reported
separately.

For the late-source generator, every input coefficient of degrees 1–16
vanishes. Each model input is therefore identical for all four base
operators and all 32 delayed factors. Under a uniform independent base
prior, a deterministic two-base proposal has expected recovery 1/2 on this
family. A particular finite roster may deviate from that expectation.

For a future-collision pair, let a base program have an output \(B(Q)\) on a
new input \(Q\), and construct an unseen mutant \(B(Q)(1+q^{224})\).
Both members have identical visible demonstrations through degree 192 and
identical query inputs. Every deterministic algorithm that sees only that
information, including one that makes public interpreter calls, produces
the same candidate on both members. A single predicted full output through
degree 256 matches **at most one** member of the pair. RRC-24 contains eight
such pairs.

## Work accounting

A full base-skeleton query evaluates both demonstrations through degree 192,
including a supplied exact algebraic suffix check. Identity, sign and
degree substitution use \(O(T)\) coefficient work; reciprocal uses an
\(O(T^2)\) triangular solve. The nominal ledger uses \(T(T+1)/2\) for
reciprocal and linear counts for other base operations and residual checks.
An exact short-prefix base-ranking control additionally evaluates all four
base maps on both degree-16 demonstrations; those eight executions receive
explicit preparation work. The neural policy pays all dense forward MACs.
A literal syntax candidate with progressive verification is charged at
each visited stage.

These are finite arithmetic and information identities; the experiment
does not prove a new Rogers–Ramanujan theorem or recover unseen algebraic
primitives. Timed CPU measurements are implementation- and host-specific.
