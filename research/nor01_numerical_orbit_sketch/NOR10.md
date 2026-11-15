# NOR-10: constructive one-bit terminal codec for two certified GRU attractors

**November 15, 2026.** This is a concrete constructive consequence of NOR-09's source-locked two-capture theorem and the merged MathCan NOR-08 label-capacity lower bound. It constructs an explicit one-dimensional terminal metric for two already certified natural-input streams using a single run-time basin bit plus one shared high-precision rational distance constant. It does not train a general basin classifier or implement OpenAI Math #094's existential finite-set embedding.

## Source evidence and the geometric lower bound

The original frozen 1,000-parameter RRC-25 trained GRU has two source-certified post-capture 24-step streams:

\[
z_1(m)=F^{1308+24m}(h_{\mathrm{signed}}),\quad
z_2(m)=F^{660+24m}(h_{\mathrm{squared}}).
\]

The independently verified nonlinear invariant balls have rational centers \(c_1,c_2\) and radii \(10^{-11},10^{-6}\). They are separated by more than \(5.020911335789842\) in true Euclidean distance. NOR-09 supplies exact-rational monotone tail bounds

\[
\|z_1(m)-c_1\|_2\le B_1(m),\qquad
\|z_2(m)-c_2\|_2\le B_2(m),
\]

where \(B_i(m)\) includes the rigorously bounded distance from the known rational center to the true fixed point.

The first stream is certified to enter its invariant ball after 109 original 12-step blocks. The second stream enters the other after 55 blocks. Both sources use only the original nine public input-feature rows, followed by a declared zero-input recurrence.

## Exact rational distance quantizer

Let

\[
D=\|c_1-c_2\|_2,\quad
Q=\frac{\lfloor 10^{18}D\rfloor}{10^{18}}.
\]

The script [nor10_binary_terminal.py](nor10_binary_terminal.py) computes \(D^2\) exactly from both stored rational center vectors. To find the integer floor without floating-point rounding, it evaluates

\[
n=\left\lfloor\sqrt{
  \left\lfloor \frac{D^2_{\mathrm{num}}10^{36}}
                       {D^2_{\mathrm{den}}}\right\rfloor}
  \right\rfloor
\]
using Python's integer square root. The construction checks

\[
\boxed{Q^2\le D^2\le(Q+10^{-18})^2,}
\qquad
\boxed{|D-Q|\le10^{-18}.}
\]

Now use a **binary terminal code** \(b\in\{0,1\}\), with decoded scalar vector

\[
\phi(0)=0,\qquad \phi(1)=Q\in\ell_p^1.
\]

For every \(1\le p<\infty\), the encoded distance between the two basin labels is exactly \(Q\), since the \(\ell_p\) norm of a one-dimensional vector is its absolute value. The scalar \(Q\) is **global codebook metadata** stored once; each known late state requires one bit to identify its basin.

### Theorem: all-late metric guarantee for the explicit codec

Fix a retained-prefix cutoff \(M\). All subsequent 24-step sampled states of the two previously captured streams have center errors bounded by \(B_i(M)\). If two states belong to the same basin \(i\), their true distance is at most \(2B_i(M)\), while their common terminal code has zero distance. If the states belong to different basins, the reverse triangle inequality gives

\[
\bigl|\|z_1(m)-z_2(n)\|_2-D\bigr|
\le B_1(M)+B_2(M)
\]

for all \(m,n\ge M\). Replacing \(D\) by the rational quantization \(Q\) costs at most \(10^{-18}\). Therefore the **explicit one-bit code** satisfies

\[
\boxed{
E_{\rm terminal}(M)
\le\max\left\{
2B_1(M),\;
2B_2(M),\;
B_1(M)+B_2(M)+10^{-18}
\right\}.
}\tag{1}
\]

This is a constructive one-dimensional \(\ell_p\) tail representation, checked using exact rational receipt arithmetic.

The obstruction is equally explicit. Since every cross-basin pair of post-capture 24-step states remains more than \(5.020911335789842\) Euclidean units apart, assigning **zero bits** with one shared terminal code incurs greater than that much additive cross-basin distortion. At the certified lower distortion budgets of NOR-09, one basin bit is necessary. The construction above is therefore optimal in terminal label cardinality for these **two specifically certified streams**, with its shared rational metric constant treated as codebook metadata.

## Boundaries

The codec applies to the post-capture 24-step samples of the two particular public-source trajectories with already proven basin membership. The labels signed and squared are verified source identities for this construction. A classifier for other public inputs must infer or certify their basin membership separately; the 72-case NOR-08C study found substantial overlap of both recurrent modes across source families.

Finite-prefix compression, the full unstrided intermediate recurrent states, and accuracy on native RRC-25 theorem-related tasks remain separate goals. For finite prefixes, NOR-09 still supplies the existential real \(L_p\) dimension reduction via OpenAI Math #094, with an explicit finite-pool embedding error \(E_M\); the simple one-bit code covers only the infinite late tails.
