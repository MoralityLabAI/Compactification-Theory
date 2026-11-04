"""QO-08: paired stream integrity and predeclared-budget pilot checks."""
import unittest
from fractions import Fraction as Q
from itertools import product

from research.ingression_dynamics_v1.paired_probe_pilot import (
    SCENARIOS, SEEDS, POLICIES, CHECKPOINTS, stream_seed,
    deterministic_cell_prefix, run_policy, verify_common_stream_prefix,
    build_report,
)
from research.ingression_dynamics_v1.anytime_active_quotient import (
    certificate, ProbeLedger,
)
from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k


class PairedProbeAudit(unittest.TestCase):
    def test_predeclared_protocol_and_true_labels(self):
        self.assertEqual(POLICIES, ("round_robin", "width", "bottleneck"))
        self.assertEqual(SEEDS, (11, 19, 23, 37, 41, 53))
        self.assertEqual(CHECKPOINTS, (24, 64, 128))
        self.assertEqual(len(SCENARIOS), 3)
        ids = []
        for case in SCENARIOS:
            self.assertLessEqual(len(case["rates"]), 8)
            value, _, labels = optimum_for_k(case["rates"], case["classes"])
            self.assertEqual(len(labels), len(case["rates"]) + 1)
            ids.append(case["id"])
            self.assertIsInstance(value, Q)
        self.assertEqual(len(ids), len(set(ids)))

    def test_cell_streams_paired_across_policies(self):
        case = SCENARIOS[1]
        for seed in (11, 23):
            for i, a in product(range(len(case["rates"])),
                                range(len(case["rates"][0]))):
                prefix = deterministic_cell_prefix(1, seed, case["rates"], i, a, 23)
                self.assertEqual(prefix[:10],
                    deterministic_cell_prefix(1, seed, case["rates"], i, a, 10))
                self.assertEqual(len(prefix), 23)
                self.assertEqual(stream_seed(1, seed, i, a),
                                 stream_seed(1, seed, i, a))

    def test_replay_each_policy_and_same_query_budget(self):
        case = SCENARIOS[1]
        for policy in POLICIES:
            one, receipts = run_policy(1, case, 19, policy,
                                       checkpoints=(0, 7, 16))
            two, reread = run_policy(1, case, 19, policy,
                                     checkpoints=(0, 7, 16))
            self.assertEqual(one, two)
            self.assertEqual(receipts, reread)
            self.assertEqual(len(receipts), 16)
            self.assertTrue(verify_common_stream_prefix(1, case, 19, receipts))
            self.assertEqual([row["queries"] for row in one], [0, 7, 16])
            self.assertEqual([sum(map(sum, row["count_by_cell"]))
                              for row in one], [0, 7, 16])
            self.assertEqual(one[0]["decision"], "undetermined")
            self.assertEqual(one[0]["robust_upper"], "1/2")

    def test_evidence_only_selection_and_no_lookahead(self):
        case = SCENARIOS[0]
        identical_cases = (
            case,
            {**case, "rates": tuple(tuple(Q(1)-x for x in row)
                                   for row in case["rates"])}
        )
        # At zero observations, policies choose from the public schema and
        # the current ledger. Different private rates can change outcomes
        # subsequently, but the initial ledger/certificate is identical.
        for c in identical_cases:
            result = certificate(
                ProbeLedger(len(c["rates"]),len(c["rates"][0])).snapshot(),
                c["classes"],c["tolerance"])
            self.assertEqual(result["decision"], "undetermined")
            self.assertEqual(result["robust_macro_error_upper"], Q(1, 2))

    def test_small_pilot_includes_all_prespecified_rows(self):
        cases = SCENARIOS[:2]
        seeds = SEEDS[:2]
        budgets = (0, 8, 16)
        report = build_report(
            scenarios=cases, seeds=seeds,
            policies=POLICIES, checkpoints=budgets)
        self.assertEqual(report["run_counts"], {
            "policy_trajectories": 12,
            "snapshots": 36,
            "queries_all_trajectories": 192,
            "matched_stream_prefix_verified": 12,
        })
        self.assertEqual(len(report["groups"]), 18)
        self.assertEqual(len(report["paired_deltas"]), 12)
        self.assertEqual(len(report["trial_rows"]), 36)
        self.assertTrue(all(row["queries"] in budgets
                            for row in report["trial_rows"]))
        self.assertTrue(all(not row["false_cert"] or
                            not row["true_rates_covered"]
                            for row in report["trial_rows"]))
        self.assertEqual(report, build_report(
            scenarios=cases, seeds=seeds,
            policies=POLICIES, checkpoints=budgets))

    def test_paired_comparisons_reproduce_from_raw_trials(self):
        report = build_report(scenarios=SCENARIOS[:1],
                              seeds=SEEDS[:3], checkpoints=(12, 24))
        for entry in report["paired_deltas"]:
            a = [r for r in report["trial_rows"] if
                 r["scenario"] == entry["scenario"] and
                 r["queries"] == entry["budget"]]
            by = {(r["seed"], r["policy"]): Q(r["true_frozen_error"])
                  for r in a}
            delta = sum((by[(seed, entry["baseline"])] -
                         by[(seed, "bottleneck")] for seed in SEEDS[:3]),
                        Q(0)) / 3
            self.assertEqual(Q(entry["paired_mean_true_error_improvement"]),
                             delta)


if __name__ == "__main__":
    unittest.main()
