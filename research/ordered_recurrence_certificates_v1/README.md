# OVC-01: ordered verification and recurrence compilation

Started November 14, 2026. New source pair: OpenAI Mathematics [#116](https://github.com/openai/math/blob/main/lean/docs/116.md), *One Rational Matrix Hitting Point for Noncommutative Formulas*, and [#135](https://github.com/openai/math/blob/main/lean/docs/135.md), *Homogeneous depth-five lower bounds for iterated matrix multiplication*.

[Theory and proof boundaries](THEORY.md) | [Exact reference](reference.py) | [Independent tests](test_reference.py)

The source papers suggest a verification contract for exact recurrent-program compression. The ordered matrix-product target has a compact streaming recurrence. A candidate formula can be checked using a single source-derived matrix substitution with a dimension bound depending on its representation size. The scalar specialization of a correct homogeneous depth-five compilation is subject to the source #135 lower bound. This is a conditional statement about exact algebraic programs in specified models.

## Finite results implemented here

The rational hitting matrices from #116 are executed through an O(D)-operation recurrence for their Taylor-coefficient action. Exact word expansion and dense matrices supply independent oracles. The finite benchmark has six valid rewrites and six scalar-valid, order-breaking rewrites. It also checks the full published formula dimension on three cases.

An elementary expansion proves that a single adjacent swap of unequal symbols in a length-k operator word first appears in Taylor degree k+1. Dimension k+2 therefore detects this restricted mutation class. All 3,282 cases over three variables at lengths two through six are checked exactly.

The source probe's trace sees only the polynomial's constant term. The runner tests the full first column rather than replacing it with that lossy scalar observation. Explicit truncation controls show how a small zero probe can miss a nonzero high-degree word.

For iterated matrix multiplication, a prefix/suffix coefficient minor is the identity. This certifies exact separable cut width n at each internal layer. Fifteen internal cuts for n=2,...,6 are checked, and numeric streaming evaluation is compared with independent path enumeration.

## Reproduce

Python 3.10+ standard library only:

    python -m unittest discover -s research/ordered_recurrence_certificates_v1 -p 'test_*.py' -v
    python research/ordered_recurrence_certificates_v1/reference.py --out /tmp/ovc01_exact.json

All executed arithmetic uses exact Fractions. Source-derived completeness is explicitly labeled conditional. A nonzero matrix witness certifies an incorrect ordered rewrite independently of the original general hitting theorem. A zero short-probe output requires the independent finite word oracle or a justified size theorem.

## Applied experiment contract

A future compact-reasoner or RefutationBench task should keep the original ordered program host-owned, let the agent propose a bounded-size replacement, and score exact semantics separately from representation cost. Record formula-tree size, any shared DAG size, the probe dimension and exact arithmetic heights. Source expressions and the evaluation interface cannot be changed by the candidate. Compare safe reassociation/distribution with order-changing mutants. A trace-only output or uncharged retained copy of the original expression does not satisfy the contract.

The current round evaluates finite algebraic programs. It makes no claim about trained transformer performance, approximate distillation, finite-field circuit consensus, or production BitVM safety. The source #135 bound is asymptotic with an unspecified sufficiently-large threshold and is not used as a small-n numerical lower bound. Previous NHB work remains on its separate branches.
