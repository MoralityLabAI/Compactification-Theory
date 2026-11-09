# RRC-12: graph classes, finite certificates and work bounds

## Graph semantics

Let \(\mathcal U=1+q\mathbb Z[[q]]\). For \(F\in\mathcal U\), use four operations:

\[
I(F)=F,\quad N(F)(q)=F(-q),\quad S(F)(q)=F(q^2),\quad R(F)=F(q)^{-1}.
\]

For graph \(g=(L,Q,P)\in\{I,N,S,R\}^3\), define
\[
E_g(F)=P\bigl(L(F)\cdot Q(F)\bigr).
\]

Formal inversion is defined because the constant term is 1. Product coefficients and reciprocal coefficients are integers and depend only on coefficients of the input through the same degree. Hence \(E_g(F)\bmod q^{T+1}\) is computable exactly from \(F\bmod q^{T+1}\). This proves that degree-\(T\) demo matching is a well-defined finite certificate with an exact verifier.

## Graph aliases and supervision

The graph syntax contains evident aliases. Branch multiplication is commutative, so \((L,Q,P)\) and \((Q,L,P)\) agree universally. Also, \(F\cdot F^{-1}=1\), \(N^2=I\), \(R^2=I\), and \(N(F(q^2))=F(q^2)\). Other graph-specific equalities follow from homomorphism properties of \(N\) and \(S\). RRC-12 groups all 64 graph strings by execution on eight fixed integer probes, yielding **25 observed signatures**. Every claimed train/test distinction means disjointness under those observed signatures. Ten independent input probes detect no collisions among the 25 representatives, supporting this implementation convention. These checks do not prove that all 25 signatures are pairwise distinct for *all* formal series, nor that all within-signature aliases hold universally.

Training outputs the lexicographically first graph string for each finite-probe signature, giving a deterministic graph-token target even when many syntax strings fit the demonstrations. Evaluation credits **any** syntax graph passing full observed-prefix execution, regardless of which token string the source used. This avoids treating equivalent programs as errors, but model training remains teacher-forced at the token level.

## Exact search versus learned proposal

For observed two demonstrations \((F_1,G_1),(F_2,G_2)\), define
\[
V_T(g)=\bigwedge_{i=1}^2\left(E_g(F_i)\equiv G_i\pmod{q^{T+1}}\right).
\]

The exhaustive method enumerates all 64 graph strings, accepting the first with \(V_{12}(g)\). It is sound and complete **for this finite graph language and observed coefficients**. The one-proposal neural model chooses \(\hat g\) from raw observations and then calls the same exact verifier exactly once. It can abstain through verifier rejection. A preview method evaluates all 64 programs to degree four, sorts them by deterministic residual statistics, then verifies four at degree 12. The preview's candidate executions and parameter-free ranking are computational resources, separate from neural weights.

A direct model with a correct proposal can use about one full graph verification. Its accuracy depends on predicting a program from the examples, while symbolic search uses the hand-supplied interpreter in choosing the candidate. The two resource vectors are **not** a common scalar unless integer arithmetic, floating-point MACs, memory and preprocessing are priced by an explicit platform model.

## Information-theoretic future ambiguity

Take any \(K>T\). A target series \(G\) and a mutant \(\widetilde G=G+q^K\) have identical observed coefficients through degree \(T\). A deterministic predictor seeing those coefficients and an unchanged query therefore produces the same candidate future on both. Because their true degree-\(K\) values differ, at most one candidate future can be correct. RRC-12 supplies eight base/mutant pairs with \(T=12,K=15\). The exact solver accepts both observed prefixes and correctly predicts at most eight of those sixteen possible futures. A model that abstains on both has no false acceptances and obtains no future-correct credits.

## Declared operation proxies

With two length-13 demonstrations, four features per degree, a one-layer GRU of hidden width \(h=32\) performs approximately
\[
C_{\rm GRU}(r)=r\,(26)(3)(32)(4+32)+12\cdot32
=89{,}856r+384
\]
FP multiply-accumulates for \(r\) complete observation scans, excluding sigmoid/tanh and data normalization. The feedforward hidden width 34 uses \(104\cdot34+34\cdot12=3{,}944\) such operations. Its 3,990 parameters are 1.35% fewer than the 4,044 recurrent parameters. The GRU carries 32 FP32 hidden values (1,024 bare state bits).

RRC-11's interpreter charges the deliberately conservative coefficient-work proxy \(3(T+1)^2\) per candidate/demo, even when unary operations are linear. RRC-12 reports all preview and verification costs in this same unit, plus dense neural MACs in their own column. The proxy omits big-integer bit complexity, Python overhead, source construction, memory moves and model training cost.
