"""Checks of the G1 governance-state compactification (about 45 s on one CPU core)."""
import unittest

from compaction import ACTIONS, CODES, all_keys, analyse, fast_witness, from_key, key_of, transition_table
from finite_model import governance_system, quotient_witness


class TestFastWitness(unittest.TestCase):
    def test_agrees_with_pairwise_reference_on_g0_system(self):
        system = governance_system()
        approval = dict(system.observations)
        exact = {s: s for s in system.states}
        self.assertIsNone(fast_witness(system, exact))
        self.assertEqual(fast_witness(system, approval) is None, quotient_witness(system, approval) is None)


class TestCouncilCompaction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keys = all_keys()
        cls.table = transition_table(cls.keys)
        cls.result = analyse(cls.keys, cls.table)
        cls.cases = {(c["alphabet"], c["observation"]): c for c in cls.result["cases"]}

    def test_table_is_closed(self):
        keys = set(self.keys)
        self.assertEqual(len(self.table), len(self.keys) * len(ACTIONS))
        self.assertTrue(set(self.table.values()) <= keys)

    def test_g0_code_exact_only_under_g0_alphabet(self):
        self.assertTrue(self.cases[("g0", "decision")]["codes"]["g0_three_bit"]["exact"])
        membership = self.cases[("membership", "decision")]["codes"]["g0_three_bit"]
        self.assertFalse(membership["exact"])
        self.assertTrue(membership["witness"]["action"].startswith("up_"))

    def test_forgeries_do_not_add_state(self):
        for obs in ("decision", "decision_dissent"):
            self.assertEqual(self.cases[("g0", obs)]["exact_classes"], self.cases[("g0_forgery", obs)]["exact_classes"])

    def test_per_member_code_is_the_minimal_exact_quotient(self):
        case = self.cases[("full", "decision")]
        code = case["codes"]["per_member_yes"]
        self.assertTrue(code["exact"])
        self.assertEqual(code["symbols"], case["exact_classes"])
        self.assertEqual(code["random_traces"]["diverged"], 0)

    def test_dissent_observable_needs_the_full_key(self):
        case = self.cases[("full", "decision_dissent")]
        self.assertEqual(case["exact_classes"], len(self.keys))
        self.assertFalse(case["codes"]["per_member_yes"]["exact"])
        self.assertEqual(case["codes"]["full_key"]["random_traces"]["diverged"], 0)

    def test_coarse_codes_diverge_on_engine_traces(self):
        case = self.cases[("full", "decision")]
        for name in ("approval_bit", "g0_three_bit", "vote_counts"):
            self.assertFalse(case["codes"][name]["exact"])
            self.assertGreater(case["codes"][name]["random_traces"]["diverged"], 0)

    def test_witnesses_replay_in_the_engine(self):
        """Re-run each reported successor witness from scratch in fresh engines."""
        for case in self.result["cases"]:
            for name, code in case["codes"].items():
                w = code.get("witness")
                if not w or w["kind"] != "successor" or name == "approval_bit":
                    continue
                left, right = tuple(map(tuple, (w["left"], w["right"])))
                left = (left[0], left[1], tuple(left[2]), tuple(left[3]))
                right = (right[0], right[1], tuple(right[2]), tuple(right[3]))
                enc = CODES[name]
                self.assertEqual(enc(left), enc(right))
                out = []
                for k in (left, right):
                    c = from_key(k)
                    c.process(ACTIONS[w["action"]](c))
                    out.append(enc(key_of(c)))
                self.assertNotEqual(out[0], out[1])


if __name__ == "__main__":
    unittest.main()
