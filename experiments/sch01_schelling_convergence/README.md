# SCH-01: Schelling Convergence Hypothesis

**Status: deterministic single-coder fixture, 2026-10-30.** This protocol compares stories of technological singularity, Abrahamic eschatology, and philosophical return. Its home in Compactification-Theory is a question about information loss under semantic projection: at which level of abstraction do incompatible stories appear identical?

## Source framework
Robert M. Geraci's 2008 paper and 2010 *Apocalyptic AI* already examine the inheritance of Jewish/Christian apocalyptic motifs in AI futurism. Kurzweil's singularity is a forecast for a technological transformation around 2045. Jehovah's Witnesses describe divine intervention, future resurrection, and restored embodied life on Earth. Plotinus describes procession and reversion, Śaṅkara teaches liberating nondual recognition, and Daoist schools differ in their meanings of return. See SOURCE_LEDGER.md for anchors. All fixture prose is researcher-authored paraphrase, never a quotation.

## Exact finite lemma
Consider a three-state directed path P with vertices d -> t -> e (deficiency, transition, altered condition). Its transition vertex bears an agency label, and its states carry a six-coordinate annotation:
(time, agency, identity, ultimate, destination, authority).

Let q erase those annotations while retaining the directed path. Construct P1 with agency(t)=technology and P2 with agency(t)=divine_intervention. Then q(P1)=q(P2)=P. Every graph automorphism of this directed path fixes the unique source d, sink e, and midpoint t, hence no label-preserving isomorphism P1 -> P2 exists. **QED.** The quotient is noninjective; shared path form cannot identify agency.

This establishes an information-loss theorem on a restricted toy class. It is neither a theorem about world history nor about the existence of God.

## Encoding and interpretive limits
SCH-01's fixture supplies eight annotated narratives: Kurzweil, Jehovah's Witnesses, Q 14:48, Plotinus, Advaita, Daoism, a hand-authored software migration, and a hand-authored stable process. Each receives a stage path and exactly six categorical labels. “Coarse match” means both contain D -> T -> E. “Faithful match” requires exact agreement on all six labels. “Typed agreement” is the fraction of matching axes conditioned on a coarse match; it is not a metric, likelihood, or evidential probability.

The Plotinus, Advaita, Daoist, and Qur'anic rows explicitly flag their DTE projection as researcher interpretation. Converting an ontological relation or liberating recognition into future-history stages would mistake the abstraction for a native doctrine. The mundane migration is a negative control because it shares DTE while lacking religious eschatology.

## Rival explanations
- **H-transmission:** specific motifs spread through documented textual and cultural transmission. Reconstruct publication chronology, translations, direct citations, and authors' intellectual networks; Geraci is prior work, not an independent discovery by this project.
- **H-cognition:** humans repeatedly favor stories of deficiency, transformation, and renewal. Compare blinded annotations with length- and genre-matched mundane and literary controls and unrelated traditions.
- **H-metaphysical:** independently derived, narrow correspondences may support a common underlying ontology. This requires predictions and likelihoods specified in advance that transmission and cognition explain poorly. No Bayes factor can presently be computed.

These explanations can overlap. The recorded corpus offers no empirical evidence identifying the metaphysical explanation.

## Full preregistration and stop rules
1. Register source editions, languages, exact passage boundaries, timestamps, translations, and provenance. Use passages as units rather than entire religions.
2. Write an independent taxonomy/codebook including unknown, disputed, and multi-label annotations. Distinguish future intervention, ontological reversion, atemporal recognition, and cyclical cultivation.
3. Create a dated influence graph with supported, plausible, and unknown connections. Claims of independence require an explicit absence-of-evidence search and appropriately limited conclusions.
4. Obtain independent blinded annotations from two qualified coders per passage; record disagreements, adjudication, per-axis agreement, and translation uncertainties.
5. Include adversarial controls matched for text length, rhetorical genre, and projected DTE shape. Reserve a held-out sample before choosing similarity thresholds.
6. Report coarse and label-aware analyses separately. Pre-register null distributions, permutation units, ablations, and correction for multiple comparisons.
7. Restrict each conclusion to its evidence grade. A theorem about the information lost in a quotient is distinct from a numerical pattern in texts or a theological argument.

**STOP:** Report no historical independence, statistical convergence, or metaphysical Bayes factors from the present single-coder fixture. Do not change the doctrinal ontology after inspecting a favorable score without documenting an exploratory revision.

## Run
Only Python's standard library is required.

    cd experiments/sch01_schelling_convergence
    python -m unittest discover -s . -p 'test_*.py' -v
    python sch01.py --output results/fixture_smoke.json

The initial hand-annotated fixture has 8 rows and 28 unordered pairs; 21 share the coarse DTE motif and 0 agree on all six axes. Kurzweil and JW agree in 1/6 axis labels (future discontinuity). Kurzweil and a **synthetic software migration** agree on 2/6. These numbers are a reproducible illustration of the researcher's labels, not a cross-cultural estimate.

## Manuscript relationship
INTERLUDE_PROPOSAL.md supplies an unnumbered Chapter 14 inset for *Divine Elegance: Muʿtaziliya After Gödel*. The separate v2.4 candidate is left unchanged, along with the nineteen chapters, seven RSI interludes, and E0–E8. E0 maintains the creator/creation boundary, E6 requires epistemic provenance, and E7 requires authority to be separately justified.
