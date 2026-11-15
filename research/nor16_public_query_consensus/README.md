# NOR-16: Public query consensus beyond the collapsed recurrent representation

Protocol registered 15 November 2026, **before opening the new held-out source cohorts**. This follows [NOR-15A/B #164](https://github.com/MoralityLabAI/Compactification-Theory/pull/164), [NOR-15C #167](https://github.com/MoralityLabAI/Compactification-Theory/pull/167), [NOR-15D #169](https://github.com/MoralityLabAI/Compactification-Theory/pull/169), and formal [MathCan NOR-15D #86](https://github.com/patrickdugan/MathCan/pull/86).

## Scientific question

The original frozen 1000-parameter RRC-25 GRU receives a 9×8 tensor extracted from the earliest source coefficients, while the public task supplies two observed input/output demonstrations through degree 192 **and the query INPUT coefficients through degree 256**. The query's *output* through degree 256 is withheld for independent grading.

NOR-15C found that no fixed shared four-check candidate order could reach more than 15/40 future-correct on its 40 aliased late_source cases. NOR-15D showed that a selector observing all sixteen public accepted-template bits, and choosing one candidate per acceptance mask, has a hindsight upper bound of 25/40. Neither bound applies to a candidate-ranking process that also examines public query INPUT coefficients and computes hypothetical future outputs.

This experiment tests two constructive ways to extract information absent from the frozen nine-step GRU representation:

1. **Public query predictive consensus.** Simulate each of the sixteen candidate programs on the fully *public* query input. Count the candidates yielding each equal 257-coefficient hypothetical output, and rank candidates by their predicted-output cluster size. Ties retain the frozen GRU ordering. The first public-accepted template within four checks is returned.
2. **Public acceptance-conditioned predictive consensus.** Check all sixteen templates against the two public demonstrations, then choose a template from the largest exact hypothetical-output cluster among public-accepted candidates. This arm spends **sixteen public full checks** and sixteen query simulations; its comparison against a four-check ranker is explicitly unequal-budget.

The second arm has a Bayesian interpretation under the synthetic generator's uniform candidate prior. Conditioning on the complete public demos and fixed header leaves each publicly accepted U/V template with equal posterior mass. If q candidate templates share a hypothetical query output among m accepted templates, selecting that output succeeds with conditional probability q/m under this assumed generative model. The conditional maximum is the largest accepted output-cluster mass divided by m. The rule uses no future output labels in selecting a candidate.

## Frozen reproducible design

- Original trained GRU checkpoint SHA-256 \`54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b\`. The 16-class task head is immutable; the original header decoder and native public verification semantics are retained.
- Four new source-independent test cohorts, each with 48 \`late_source\` public inputs, 12 \`signed\`, and 12 \`rational\`. New source seed offset: \`corpus.SEED+93116000+rep*1000003+family_index*10007\`; source and nonlate model-tensor collisions with earlier NOR-11 through NOR-15B samples are excluded. Primary set: **192 distinct complete-public-input late_source cases sharing one GRU-visible tensor**. Secondary: 96 cases with distinct nonlate model features.
- No task weights, training labels, or development scale choices are fitted during NOR-16. All policies are fully specified before testing these four cohorts. The frozen GRU provides the original tie-breaking preference. The negative control shuffles class order with a deterministic hash of the public query input after paying for the same sixteen hypothetical candidate simulations.
- Query simulation uses four cached public unary operator evaluations on the input source, then exactly constructs all sixteen hypothetical future outputs to degree 256. It never compares them with the withheld target output. This is substantial extra public arithmetic and is separately reported from exact demonstration verification checks.
- Native four-check arms: \`original_4\` (no extra simulations), \`consensus_4\` (16 simulated public query predictions), \`hash_4\` (matched 16 predictions, ignored by the policy). Each runs **at most four full exact public candidate checks**, stopping at first acceptance.
- Sixteen-check comparator arms: \`full_mask_original_16\` (first accepted candidate in the original frozen order), and \`full_mask_map_16\` (accepted-set-conditioned maximum query-output consensus). Both check all sixteen candidate templates publicly, with final candidate acceptance reused. They are deliberately **not equal-cost** to the native four-check arms. Their query simulation cost is separately charged.
- Every arm freezes its candidate before the original grade function accesses the withheld true query output at degree 256. Test-source program identity and query output are never accessed by the selection logic. The unit test replaces the private program and all future output coefficients without changing any policy ranking, public mask or predicted-output partition.
- Public observability counterfactual: if two templates agree with both full public demonstrations yet predict unequal outputs on the same public query source, they yield two valid generative worlds with identical complete public observations and distinct future targets. Store their accepted indices, first differing output degree and public-data equality certificate, without consulting the original private target.
- Report native future-correct, false accepts, exact paid public checks, number of query simulations, exact public accepted-set sizes, candidate-output equivalence classes, posterior expected MAP correctness under the uniform program-prior assumption, and public-indistinguishability witnesses. Primary paired case sign tests are exploratory given shared family generation and are accompanied by four separate replicate results.

**Preregistered prospective primary success rule:** \`consensus_4\` must outperform both \`original_4\` and the matched \`hash_4\` on the 192 complete-public-input late_source cases by positive paired net gain and two-sided exact sign-test p≤0.025 in each comparison. The \`full_mask_map_16\` and \`full_mask_original_16\` arms are a separately metered discovery of what extra exact public checks buy; their results cannot claim native equal-budget superiority.

## Interpretation and next theorem

A posterior cluster containing q accepted programs has an exact conditional mass q/m under the declared uniform prior, which is a count-based MAP theorem. The public-data counterfactual certificates prove observation-level non-identifiability whenever two accepted programs yield divergent futures, even if a reasoner sees every observed coefficient and the entire query source input. These bounds concern the synthetic RRC25 program family and do not imply a global upper bound on arbitrary mathematical reasoning.

The central capability question is whether prospective query-output consensus improves correct proof-template choice under the same paid four-check budget. Any positive result still requires an independent seed replication and explicit accounting for extra public candidate simulations.

## Execute

    python -m unittest research.nor16_public_query_consensus.test_experiment -v
    python research/nor16_public_query_consensus/experiment.py --check \
      --repetitions 4 --late-per-repetition 48 \
      --signed-per-repetition 12 --rational-per-repetition 12 \
      --out /tmp/nor16_full.json

The first full hosted CI run will emit a source-locked JSON receipt. This protocol precedes that execution; no NOR-16 result is claimed at registration time.
