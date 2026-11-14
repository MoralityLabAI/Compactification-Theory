"""SCT-01B: SAT stream monitor resource, witness, and duplication controls."""
from __future__ import annotations
from fractions import Fraction as F
import unittest

from arena import (
    POLICIES, SEED, HORIZONS, TRIALS, decision, experiment,
    fixed_sequence, run_episode, state_bits, update,
)
from reference import hit_cdf


class SATStreamingGymTests(unittest.TestCase):
    def test_host_owns_clause_semantics_for_exact_policies(self):
        for sequence in (
            (0,)*8, tuple(range(8)), tuple(range(8,16)),
            tuple(range(16)), (0,1,0,2,3,2,1,1,1,1,1),
        ):
            a=run_episode(sequence,"exact_binary_16bit")
            b=run_episode(sequence,"certifying_16bit")
            self.assertEqual(a["ground_truth_unsat"],b["ground_truth_unsat"])
            self.assertTrue(a["correct"])
            self.assertTrue(b["correct"])
            self.assertEqual(a["mutable_state_bit_budget"],16)
            self.assertEqual(b["mutable_state_bit_budget"],16)
            self.assertFalse(b["original_clause_stream_access_at_decode"])
            if b["ground_truth_unsat"]:
                self.assertTrue(b["positive_certificate_verified"])

    def test_duplicate_clause_counter_fabricates_contradiction(self):
        sequence=(0,)*8
        for policy in ("exact_binary_16bit","certifying_16bit",
                       "first_gadget_only_8bit"):
            row=run_episode(sequence,policy)
            self.assertFalse(row["ground_truth_unsat"])
            self.assertTrue(row["correct"])
        naive=run_episode(sequence,"arrival_counters_8bit")
        self.assertTrue(naive["false_positive"])
        self.assertEqual(naive["distinct_clauses"],1)
        self.assertEqual(naive["claim"],"UNSAT")

    def test_first_gadget_monitor_can_miss_full_second_gadget(self):
        stream=tuple(range(8,16))
        first=run_episode(stream,"first_gadget_only_8bit")
        self.assertTrue(first["ground_truth_unsat"])
        self.assertTrue(first["false_negative"])
        for policy in ("exact_binary_16bit","certifying_16bit"):
            self.assertTrue(run_episode(stream,policy)["correct"])

    def test_certifying_first_core_stays_valid_after_other_core(self):
        stream=tuple(range(8,16))+tuple(range(8))
        row=run_episode(stream,"certifying_16bit")
        self.assertTrue(row["correct"])
        self.assertEqual(row["proof_gadget"],1)
        self.assertTrue(row["positive_certificate_verified"])

    def test_zero_budget_and_invalid_inputs(self):
        with self.assertRaises(ValueError): state_bits("not_a_policy")
        with self.assertRaises(ValueError): update("exact_binary_16bit",0,16)
        with self.assertRaises(ValueError): update("certifying_16bit",-1,0)
        with self.assertRaises(ValueError): run_episode((),POLICIES[0])
        with self.assertRaises(ValueError): experiment(horizons=(),trials=10)

    def test_fixed_source_seeds_are_replayable(self):
        a=fixed_sequence(SEED,128)
        b=fixed_sequence(SEED,128)
        self.assertEqual(a,b)
        self.assertEqual(len(a),128)
        self.assertTrue(all(0<=x<16 for x in a))

    def test_frozen_small_control_matches_exact_cdf_and_no_oracle_errors(self):
        a=experiment(horizons=(8,16,31,40),trials=128)
        b=experiment(horizons=(8,16,31,40),trials=128)
        self.assertEqual(a,b)
        self.assertEqual(a["status"],"FROZEN_SYNTHETIC_SAT_STREAM_CONTROL")
        self.assertEqual(len(a["cohorts"]),4)
        for cohort in a["cohorts"]:
            self.assertEqual(F(cohort["exact_UNSAT_probability"]),
                             hit_cdf(cohort["horizon"]))
            for policy in cohort["policies"]:
                self.assertLessEqual(policy["correct"],128)
                if policy["policy"] in ("exact_binary_16bit","certifying_16bit"):
                    self.assertEqual(policy["false_negative"],0)
                    self.assertEqual(policy["false_positive"],0)
                    self.assertEqual(policy["correct"],128)
                if policy["policy"]=="first_gadget_only_8bit":
                    self.assertEqual(policy["false_positive"],0)
                if policy["policy"]=="certifying_16bit":
                    self.assertEqual(policy["witness_attempts"],
                                     policy["verified_positive_witnesses"])

    def test_samples_are_independent_across_horizons(self):
        r=experiment(horizons=(24,32),trials=128)
        self.assertNotEqual(r["cohorts"][0]["sequences_sha256"],
                            r["cohorts"][1]["sequences_sha256"])
        self.assertGreater(F(r["cohorts"][1]["exact_UNSAT_probability"]),
                           F(r["cohorts"][0]["exact_UNSAT_probability"]))


if __name__=="__main__":
    unittest.main()
