# OIC-20: labeled marker sensors and generator identifiability

**Status:** exact finite-state results with standard-library code, exhaustive searches and written proofs. The labeled-marker interface adds sensors to OIC-19. It is a new experiment rather than a re-analysis of the original one-marker observations.

## Definition

Fix \(n\ge 2\), unknown constant \(d\in\mathbb Z_n\), and \(z_{t+1}=z_t+d\) with \(z_0\) uniform on \(\mathbb Z_n\). A set \(M=\{m_1,\ldots,m_s\}\subset\mathbb Z_n\) of **labeled** markers returns \(y_t=( [z_t=m_1],\ldots,[z_t=m_s])\) at every active observation. Each label denotes a distinct, sensor-addressable position. The observer knows n and M but not d or z0. Exactly n consecutive active observations contain a full period for every possible shift; equality of the distributions of their length-n strings is equivalent to equality of the complete observable process laws, including adaptive subsampling.

OIC-T26 established that a single marker identifies exactly \(g=\gcd(n,d)\).

## OIC-T27: the exact two-marker signature

Place the markers at \(M=\{0,k\}\), where \(0<k<n\). For shift d define \(g=\gcd(n,d)\) and \(p=n/g\). If \(g\nmid k\), the two markers lie in distinct d-orbits; the law is determined by g. If \(g\mid k\), both markers lie in the same orbit. The lag from a hit at 0 to a subsequent hit at k, modulo the orbit period p, is

\[
\boxed{\delta_{n,k}(d)=\frac{k}{g}\left(\frac{d}{g}\right)^{-1}\pmod{p}.}
\]

The inverse exists because \(d/g\) is a unit in \(\mathbb Z_p\). Consequently two shift laws are equal if and only if they share g and (when defined) delta.

**Proof.** Each of the g phase cosets has equal prior probability. The marked orbit is reached with probability 1/g. For g not dividing k, the markers occupy different cosets, so an initial phase yields pulses at one marker, or the other, or neither, with uniform time offset modulo p. In the same-orbit case, the two pulse trains have period p and fixed relative offset δ solving \(t d\equiv k\pmod n\). Uniform initial phase randomizes the common temporal offset, leaving δ as the exact relative invariant. It is directly observable on a full period. Different g are distinguished by the probability of seeing marker 0, equal to 1/g. QED.

## OIC-T28: a second marker resolves unit-shift orientation

If \(\gcd(k,n)=1\), then for any unit shift d the markers share a single orbit (g=1) and \(\delta=k d^{-1}\pmod n\). The map d↦δ is injective on \(\mathbb Z_n^\times\). Therefore two relatively prime marker positions identify every unit shift (including its orientation), exactly. For nonunit shifts, same-gcd ambiguity may remain.

A surprising finite witness is n=6: markers (0,2) distinguish **all** six shifts, including nonunits. These marker positions are not coprime. For n=8, at least three marker labels are needed; (0,1,2) suffices.

## OIC-T29: a divisor-sensor upper bound for all shifts

Let \(D(n)\) be the positive divisors of n and choose

\[
M=\{0\}\cup(D(n)\setminus\{n\}).
\]

This set has \(\tau(n)\) labeled sensors and distinguishes every d. For nonzero d, first recover g from the single-marker pulse probability. The marker at position g lies in the same orbit as 0, and its lag is \(\delta=(d/g)^{-1}\pmod{n/g}\). Inverting δ determines d/g and hence d modulo n. For d=0, g=n is unique. QED.

This is a constructive upper bound; minimum sensor sets can be smaller. The reference program exhaustively minimizes the number of labels for n=2,...,16, quotienting away global translation of marker positions by fixing one marker at 0.

## OIC-T30: sharp minimum number of markers for prime powers

Let \(n=p^a\), p prime, a≥1. The minimum number m of labeled marker sensors that distinguishes **every** shift d under the stated uniform-phase interface is

\[
\boxed{m=\begin{cases}a,&p=2,\\a+1,&p\text{ odd}.\end{cases}}
\]

For odd p, an attaining marker set is \(\{0,1,p,p^2,\ldots,p^{a-1}\}\). For p=2, omit the last power: \(\{0,1,2,4,\ldots,2^{a-2}\}\), with the empty-power convention giving the singleton {0} at a=1.

**Sufficiency.** For d with g=p^j, recovery of its reduced unit d/g from the lag between markers 0 and p^j is immediate whenever the reduced modulus p^(a-j) has more than one unit. The g-class is already identified by the pulse probability. For p=2 and j=a−1 the reduced modulus is 2 and contains a single unit, so no second marker at p^(a−1) is needed. QED.

**Necessity.** At each level j for which φ(p^(a−j))>1, distinguish two shifts with gcd p^j whose reduced units differ by p^(a−j−1) (take units 1 and 1+p^(a−j−1), or 1 and 2 when a−j=1 and p odd). Unless some sensor pair has difference of exact p-adic valuation j, all sensor pairs in a common shift orbit have differences divisible by p^(j+1). Their relative marker-hit lags then agree for these two reduced units, and the full observable processes coincide. Thus each such j must appear as the exact p-adic valuation of at least one marker difference.

Among m distinct p-adic points, at most m−1 different exact valuations occur across all pair differences: as the partition into congruence classes modulo p^j refines, each **distinct** valuation level corresponds to at least one split of a cluster, and there are at most m−1 splits in a hierarchy with m leaves. There are a required levels for odd p, or a−1 for p=2. Hence m≥a+1 for odd p and m≥a for p=2. The constructions attain these bounds. QED.

## Verification and scope

The exact code produces complete length-n observation-string distributions using integer counts, checks the two-marker signature for all n≤17, verifies the gcd-only one-marker law for n≤21, and exhaustively finds minimum marker counts through n=16. The prime-power formula is checked on available cases including n=16 and 27, and the divisor construction on n≤33. These finite tests are supplementary to the written proofs.

A marker label is a **sensor resource**, and recording s simultaneous bits per active time step consumes s raw observation bits. For the original OIC x-bit interface, merely positing another marker changes the task. There is no conclusion about consciousness, cosmology or an actual frontier neural network's latent topology.
