"""Exact controls and malformed-certificate checks for RefutationBench."""

import copy
import unittest

from research.refutation_bench.certificates import (
    MAX_DECLARATIONS,
    MAX_FORMULA_DEPTH,
    MAX_FORMULA_NODES,
    MAX_PREMISES,
    MAX_VARIABLES,
    CertificateError,
    dependency_closure,
    evaluate_formula,
    find_countermodel,
    minimal_entailing_subset,
    verify_certificate,
    verify_countermodel,
)


class PropositionalCertificates(unittest.TestCase):
    def test_affirming_consequent_has_verified_countermodel(self):
        problem = {
            "kind": "propositional",
            "premises": [{"implies": ["P", "Q"]}, "Q"],
            "conclusion": "P",
        }
        certificate = {"kind": "countermodel", "assignment": {"P": False, "Q": True}}
        result = verify_certificate(problem, certificate)
        self.assertTrue(result["verified"])
        self.assertEqual(result["scope"], "finite_propositional_logic")
        self.assertEqual(result["details"]["premise_values"], [True, True])

    def test_valid_modus_ponens_has_no_countermodel(self):
        premises = [{"implies": ["P", "Q"]}, "P"]
        self.assertIsNone(find_countermodel(premises, "Q"))
        self.assertFalse(verify_countermodel(premises, "Q", {"P": True, "Q": False})["verified"])
        self.assertFalse(verify_countermodel(premises, "Q", {"P": True, "Q": True})["verified"])

    def test_reversed_implication_does_not_follow(self):
        premises = [{"implies": ["P", "Q"]}]
        conclusion = {"implies": ["Q", "P"]}
        witness = find_countermodel(premises, conclusion)
        self.assertEqual(witness, {"P": False, "Q": True})
        self.assertTrue(verify_countermodel(premises, conclusion, witness)["verified"])

    def test_transitive_implication_and_its_converse(self):
        premises = [{"implies": ["P", "Q"]}, {"implies": ["Q", "R"]}]
        self.assertIsNone(find_countermodel(premises, {"implies": ["P", "R"]}))
        witness = find_countermodel(premises, {"implies": ["R", "P"]})
        self.assertEqual(witness, {"P": False, "Q": False, "R": True})

    def test_maximum_variable_count_and_last_assignment(self):
        atoms = [f"P{i:02d}" for i in range(MAX_VARIABLES)]
        conclusion = {"not": atoms[-1]}
        witness = find_countermodel(atoms, conclusion)
        self.assertEqual(witness, {atom: True for atom in atoms})
        self.assertTrue(verify_countermodel(atoms, conclusion, witness)["verified"])

    def test_vacuous_conditional_preserves_false_conclusion(self):
        premises = [{"not": "P"}, {"implies": ["P", "Q"]}]
        witness = find_countermodel(premises, "Q")
        self.assertEqual(witness, {"P": False, "Q": False})
        self.assertTrue(verify_countermodel(premises, "Q", witness)["verified"])

    def test_inconsistent_premises_cannot_supply_countermodel(self):
        premises = ["P", {"not": "P"}]
        self.assertIsNone(find_countermodel(premises, "Q"))

    def test_tautology_contradiction_and_empty_assignment(self):
        self.assertTrue(evaluate_formula({"and": []}, {}))
        self.assertFalse(evaluate_formula({"or": []}, {}))
        self.assertEqual(find_countermodel([], {"or": []}), {})
        self.assertIsNone(find_countermodel([], {"or": ["P", {"not": "P"}]}))

    def test_nested_boolean_operations(self):
        formula = {"and": [{"or": ["P", "Q"]}, {"not": {"and": ["P", "Q"]}}]}
        self.assertTrue(evaluate_formula(formula, {"P": True, "Q": False}))
        self.assertFalse(evaluate_formula(formula, {"P": True, "Q": True}))

    def test_minimum_subset_reports_indices_and_deterministic_ties(self):
        premises = ["R", {"implies": ["P", "Q"]}, "P", "P"]
        self.assertEqual(minimal_entailing_subset(premises, "Q"), [1, 2])
        self.assertEqual(minimal_entailing_subset(["P", "P"], "P"), [0])
        self.assertEqual(minimal_entailing_subset(["P"], {"or": ["Q", {"not": "Q"}]}), [])
        self.assertIsNone(minimal_entailing_subset(["P"], "Q"))

    def test_assignment_requires_exact_atoms_and_strict_bool_values(self):
        bad_assignments = [{}, {"P": True, "extra": False}, {"P": 1}, {"P": "true"}, []]
        for assignment in bad_assignments:
            with self.subTest(assignment=assignment), self.assertRaises(CertificateError):
                verify_countermodel([], "P", assignment)

    def test_malformed_formula_operators_are_rejected(self):
        malformed = [
            True, 0, None, [], "", " ", {}, {"xor": ["P", "Q"]},
            {"not": "P", "and": []}, {"implies": ["P"]},
            {"implies": ["P", "Q", "R"]}, {"and": "P"},
            {"not": ["P"]}, {"or": ["P", 1]},
        ]
        for formula in malformed:
            with self.subTest(formula=formula), self.assertRaises(CertificateError):
                find_countermodel([], formula)

    def test_depth_nodes_variables_and_premises_are_bounded(self):
        too_deep = "P"
        for _ in range(MAX_FORMULA_DEPTH):
            too_deep = {"not": too_deep}
        too_many_nodes = {"and": ["P"] * MAX_FORMULA_NODES}
        too_many_variables = {"or": [f"P{i}" for i in range(MAX_VARIABLES + 1)]}
        for formula in (too_deep, too_many_nodes, too_many_variables):
            with self.assertRaises(CertificateError):
                find_countermodel([], formula)
        with self.assertRaises(CertificateError):
            find_countermodel(["P"] * (MAX_PREMISES + 1), "P")
        with self.assertRaises(CertificateError):
            minimal_entailing_subset(["P"] * 13, "P")

    def test_hidden_invalid_branch_and_cyclic_python_input_are_rejected(self):
        with self.assertRaises(CertificateError):
            evaluate_formula({"or": [{"and": []}, None]}, {})
        cycle = {}
        cycle["not"] = cycle
        with self.assertRaises(CertificateError):
            find_countermodel([], cycle)

    def test_formula_budget_is_shared_across_problem(self):
        premise = {"and": ["P"] * 1100}
        with self.assertRaises(CertificateError):
            find_countermodel([premise], premise)

    def test_certificate_kind_and_extra_fields_are_rejected(self):
        problem = {"kind": "propositional", "premises": [], "conclusion": "P"}
        for certificate in (
            {"kind": "countermodel", "assignment": {"P": False}, "trusted": True},
            {"kind": "dependency_closure", "assignment": {"P": False}},
        ):
            with self.assertRaises(CertificateError):
                verify_certificate(problem, certificate)


class DeclarationCertificates(unittest.TestCase):
    def setUp(self):
        self.problem = {
            "kind": "dependency_graph",
            "target": "final",
            "declarations": {
                "used": {"kind": "axiom", "dependencies": []},
                "unused": {"kind": "axiom", "dependencies": []},
                "h": {"kind": "hypothesis", "dependencies": []},
                "lemma": {"kind": "theorem", "dependencies": ["used"]},
                "definition": {"kind": "definition", "dependencies": ["lemma"]},
                "final": {"kind": "theorem", "dependencies": ["definition", "h"]},
            },
        }
        self.certificate = {
            "kind": "dependency_closure",
            "used_axioms": ["used"],
            "unused_axioms": ["unused"],
            "local_hypotheses": ["h"],
        }

    def test_used_unused_and_local_hypotheses_stay_distinct(self):
        result = verify_certificate(self.problem, self.certificate)
        self.assertTrue(result["verified"])
        self.assertEqual(result["scope"], "declaration_metadata_only")
        closure = result["details"]["computed"]
        self.assertEqual(closure["reachable_declarations"], ["definition", "final", "h", "lemma", "used"])

    def test_hidden_assumption_and_mislabeled_hypothesis_fail(self):
        missing = copy.deepcopy(self.certificate)
        missing["used_axioms"] = []
        self.assertFalse(verify_certificate(self.problem, missing)["verified"])
        mislabeled = copy.deepcopy(self.certificate)
        mislabeled["used_axioms"] = ["used", "h"]
        mislabeled["local_hypotheses"] = []
        self.assertFalse(verify_certificate(self.problem, mislabeled)["verified"])

    def test_unknown_references_rejected_even_when_unreachable(self):
        self.problem["declarations"]["unused"]["dependencies"] = ["absent"]
        with self.assertRaises(CertificateError):
            dependency_closure(self.problem)

    def test_unknown_target_and_unknown_certificate_name_rejected(self):
        bad_problem = copy.deepcopy(self.problem)
        bad_problem["target"] = "absent"
        with self.assertRaises(CertificateError):
            dependency_closure(bad_problem)
        self.certificate["used_axioms"] = ["absent"]
        with self.assertRaises(CertificateError):
            verify_certificate(self.problem, self.certificate)

    def test_cycles_rejected_in_reachable_and_unreachable_components(self):
        for dependencies in ({"used": ["final"]}, {"unused": ["unused"]}):
            problem = copy.deepcopy(self.problem)
            for name, edges in dependencies.items():
                problem["declarations"][name]["dependencies"] = edges
            with self.assertRaises(CertificateError):
                dependency_closure(problem)

    def test_diamond_dependencies_are_valid(self):
        self.problem["declarations"]["final"]["dependencies"].extend(["lemma", "used"])
        self.assertTrue(verify_certificate(self.problem, self.certificate)["verified"])

    def test_axiom_target_counts_as_used(self):
        self.problem["target"] = "used"
        closure = dependency_closure(self.problem)
        self.assertEqual(closure["used_axioms"], ["used"])
        self.assertEqual(closure["local_hypotheses"], [])

    def test_duplicate_edges_certificate_names_and_bad_kinds_rejected(self):
        problem = copy.deepcopy(self.problem)
        problem["declarations"]["final"]["dependencies"] = ["h", "h"]
        with self.assertRaises(CertificateError):
            dependency_closure(problem)
        certificate = copy.deepcopy(self.certificate)
        certificate["used_axioms"] = ["used", "used"]
        with self.assertRaises(CertificateError):
            verify_certificate(self.problem, certificate)
        self.problem["declarations"]["used"]["kind"] = "proved_axiom"
        with self.assertRaises(CertificateError):
            dependency_closure(self.problem)

    def test_graph_size_limit_and_maximum_depth(self):
        declarations = {
            f"d{i}": {"kind": "theorem", "dependencies": [f"d{i + 1}"] if i + 1 < MAX_DECLARATIONS else []}
            for i in range(MAX_DECLARATIONS)
        }
        problem = {"kind": "dependency_graph", "target": "d0", "declarations": declarations}
        self.assertEqual(len(dependency_closure(problem)["reachable_declarations"]), MAX_DECLARATIONS)
        declarations["extra"] = {"kind": "axiom", "dependencies": []}
        with self.assertRaises(CertificateError):
            dependency_closure(problem)


if __name__ == "__main__":
    unittest.main()
