"""Versioned public questions. Answers require checkable work, not private thoughts."""
PACK_VERSION = "distill.packs.v1"
EXPLAIN = ("claim", "derivation", "check", "limitations")
STEP = ("what", "how", "why", "invariant", "failure_and_check")
CASE = ("answer", "worked_solution", "check", "assumptions")

# id, category, prompt
COMMON = (
    ("ontology", "foundations", "Define the objects, types, state space, observation and action interfaces. Identify the information available at each boundary."),
    ("motivation", "foundations", "State the problem and explain the scientific reason for the gym. Identify the concrete capability it tests."),
    ("first_principles", "foundations", "Derive a central mathematical relation from explicit assumptions. Give dimensions or types, and a checkable intermediate result."),
    ("objective", "foundations", "Write the objective, loss or reward and constraints. Identify the exact unit of measurement and the success criterion."),
    ("algorithm", "mechanism", "Specify pseudocode for the computation, termination and error handling. Separate the learned and fixed operations."),
    ("worked_example", "mechanism", "Construct a small new public example, solve it explicitly, and show intermediate quantities sufficient to reproduce the result."),
    ("verification", "evidence", "Specify an independent verification procedure and a certificate or executable check. State which part of the conclusion it verifies."),
    ("complexity", "mechanism", "Derive time, memory, parameter and communication costs as applicable. Distinguish asymptotic bounds from measured costs."),
    ("counterfactual", "evidence", "Change one premise or intervention. Predict its effect and show a derivation or an explicit counterexample."),
    ("controls", "evidence", "Design matched baselines and sham controls. Explain which confound each removes, including resource and information budgets."),
    ("failure", "evidence", "Construct a failure case and propose the smallest reproducible diagnostic. Explain how to distinguish competing causes."),
    ("leakage", "evidence", "Specify public inputs, label custody, lineage splits and leakage checks. Identify what would invalidate the evidence."),
    ("causal", "evidence", "Give an intervention, a causal claim, and the observations that could falsify it. Separate association from intervention evidence."),
    ("compactification", "transfer", "State what information or state must be retained to preserve the required behavior. Derive a bound or exhibit a collision under weaker retention."),
    ("generalization", "transfer", "Identify a changed distribution or interface; explain which guarantees survive and propose a separate held-out test."),
    ("transfer", "transfer", "Map the central mechanism into another mathematical or computational gym. Describe the shared structure and the limits of the analogy."),
    ("research_taste", "research", "Propose a cheapest decisive follow-up experiment. Give its hypothesis, alternatives, controls, compute budget, stop rule and promotion gate."),
    ("calibration", "research", "Classify the claims as derived, independently verified, observed or conjectural. Name unresolved premises and the evidence that would change your conclusion."),
)
QUICK = frozenset(("ontology", "first_principles", "objective", "algorithm", "verification", "failure", "research_taste", "calibration"))
PACKS = {
    "formal": (
        ("proof_scope", "Write the theorem with all hypotheses and quantifiers, then identify the exact proof obligations, dependencies and scope."),
        ("proof_repair", "Give a plausible invalid proof step, explain its defect with a counterexample or failed premise, and supply a checkable repair."),
        ("kernel_boundary", "Explain what the trusted checker must verify and how to avoid circular source dependencies or unproved assumptions."),
    ),
    "recurrent": (
        ("state_update", "Define the recurrent state update and unrolling. Derive a stability or stopping condition, with explicit assumptions."),
        ("matched_compute", "Design a matched-visit, token, FLOP or CPU comparison. Explain what is held fixed and what varies."),
        ("state_sufficiency", "Give two histories that do or do not require different states to answer a future query; derive a memory requirement."),
    ),
    "causal_geometry": (
        ("fiber", "Define the readout map, null space or quotient and explain which invariants hold under intervention."),
        ("dose", "Specify an intervention and a norm-matched sham, derive the effect measure and identify a nonlinear failure of a linear argument."),
        ("causal_bound", "Give a checkable lower bound, upper bound or explicit witness for observable distinctions."),
    ),
    "strategy": (
        ("strategy_value", "Derive the achievable value or regret under the stated information and action constraints."),
        ("belief", "Describe the information set, sufficient belief or hidden state, and how it changes after an observation."),
        ("policy_comparison", "Construct an exact small policy comparison and explain what resource or information restrictions make it fair."),
    ),
    "research": (
        ("information_gain", "Compare two experiments using a defined information or decision-value objective and an explicit cost."),
        ("negative_result", "Explain a null or negative result, derive its inferential limit and give a discriminating follow-up."),
    ),
    "authenticated": (
        ("authority", "Define the trusted inputs, authorization boundary and independently checkable semantics. Explain what a signature establishes."),
        ("binding", "Specify the identifiers that must be bound to the query and result; construct a replay or substitution failure and the check that rejects it."),
        ("exactness", "Explain how to verify exact values and canonical serialization independently of any candidate accelerator."),
    ),
    "algebra": (
        ("algebraic_equivalence", "Give a precise equivalence relation on candidate programs or representations; show how to test it on finite data."),
        ("identifiability", "Construct two hypotheses that agree on observations and disagree on a future query, or prove separation on a specified finite set."),
    ),
}
