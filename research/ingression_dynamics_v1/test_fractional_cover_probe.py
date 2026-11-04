"""QO-09: exact LP provenance, matched streams and reproducible pilot checks."""
import unittest
from fractions import Fraction as Q
from itertools import product

from research.ingression_dynamics_v1.fractional_cover_probe import (
    SCENARIOS, ORIGINAL_SCENARIOS, FRESH_SCENARIOS, POLICIES,
    SEEDS, CHECKPOINTS, fractional_plan, verify_fractional_design,
    choose_fractional_cell, run_fractional_policy, build_pilot_report,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as original_run_policy, verify_common_stream_prefix,
)


def independent_half_grid_lp_objective(masks, num_cells):
    """Direct LP scan at 0, 1/2, 1 (independent of QI-03 certificate)."""
    best=None
    for allocation in product((Q(0),Q(1,2),Q(1)),repeat=num_cells):
        if all(sum((allocation[i] for i in range(num_cells)
                    if mask & (1<<i)),Q(0))>=1 for mask in masks):
            cost=sum(allocation,Q(0))
            if best is None or cost<best:
                best=cost
    return best


class FractionalCoverProbeTests(unittest.TestCase):
    def test_closed_form_qi03_primal_dual_matches_independent_lp_grid(self):
        # Includes the eight-cell panel even though the original EvalMath
        # vertex oracle is capped at six cells. Blockwise primal/dual
        # feasibility is a global optimality certificate by MOR-08.
        for total in range(1,9):
            for rotated in (False,True):
                plan=fractional_plan(total,1,rotated)
                self.assertEqual(plan["cell_count"],total)
                self.assertEqual(
                    verify_fractional_design(plan["alternative_masks"],
                                              plan["primal_weights"],
                                              plan["dual_weights"]),
                    plan["exact_lp_objective"])
                self.assertEqual(
                    plan["exact_lp_objective"],
                    independent_half_grid_lp_objective(
                        plan["alternative_masks"],total))
                self.assertEqual(sum(plan["normalized_shares"],Q(0)),Q(1))
                self.assertTrue(all(x>0 for x in plan["primal_weights"]))

    def test_exact_triangle_dual_and_rotated_null(self):
        p=fractional_plan(3,2)
        self.assertEqual(p["primal_weights"],
                         (Q(1,2),Q(1,2),Q(1,2),Q(1),Q(1),Q(1)))
        self.assertEqual(p["exact_lp_objective"],Q(9,2))
        self.assertEqual(p["dual_weights"],
                         (Q(1,2),Q(1,2),Q(1,2),Q(1),Q(1),Q(1)))
        q=fractional_plan(3,2,True)
        self.assertEqual(q["primal_weights"],
                         (Q(1),Q(1),Q(1),Q(1,2),Q(1,2),Q(1,2)))
        self.assertEqual(p["exact_lp_objective"],q["exact_lp_objective"])
        self.assertNotEqual(p["alternative_masks"],q["alternative_masks"])
        r=fractional_plan(4,2)
        self.assertEqual(r["exact_lp_objective"],Q(13,2))

    def test_fixed_weighted_fair_scheduler_never_starves_cells(self):
        for n,d in ((3,2),(4,2)):
            for rotated in (False,True):
                plan=fractional_plan(n,d,rotated)
                counts=[[0]*d for _ in range(n)]
                first=[]
                for step in range(128):
                    i,a=choose_fractional_cell(counts,plan)
                    if step<n*d:
                        first.append((i,a))
                    counts[i][a]+=1
                self.assertEqual(len(set(first)),n*d)
                self.assertEqual(sum(sum(row) for row in counts),128)
                self.assertTrue(all(x>=1 for row in counts for x in row))
                weights=plan["primal_weights"]
                normalized=[Q(counts[i//d][i%d])/weights[i]
                            for i in range(n*d)]
                self.assertLessEqual(max(normalized)-min(normalized),Q(2))

    def test_rejected_bad_certificate_and_malformed_counts(self):
        p=fractional_plan(2,2)
        with self.assertRaises(ValueError):
            verify_fractional_design((0,),p["primal_weights"],(Q(1),))
        with self.assertRaises(ValueError):
            verify_fractional_design(p["alternative_masks"],
                                     p["primal_weights"],tuple(Q(0) for _ in p["dual_weights"]))
        with self.assertRaises(ValueError):
            choose_fractional_cell([[0,0]],p)
        with self.assertRaises(ValueError):
            choose_fractional_cell([[0,0],[-1,0]],p)
        with self.assertRaises(ValueError):
            fractional_plan(0,2)

    def test_candidate_family_is_public_dimension_only(self):
        self.assertEqual(len(SCENARIOS),len(ORIGINAL_SCENARIOS)+2)
        self.assertEqual(POLICIES[:3],("round_robin","width","bottleneck"))
        self.assertEqual(SEEDS,(11,19,23,37,41,53))
        self.assertEqual(CHECKPOINTS,(24,64,128))
        for case in SCENARIOS:
            n,d=len(case["rates"]),len(case["rates"][0])
            p=fractional_plan(n,d)
            self.assertEqual(p["primal_weights"],
                             fractional_plan(n,d)["primal_weights"])
            # Rate changes cannot alter the fixed LP alternative family.
            self.assertEqual(p["alternative_masks"],
                             fractional_plan(n,d)["alternative_masks"])

    def test_matched_stream_integrity_and_old_baseline_unchanged(self):
        case=SCENARIOS[1]
        checkpoints=(0,7,16)
        for policy in ("fractional_cover","fractional_cover_rotated"):
            rows,receipts=run_fractional_policy(1,case,19,policy,checkpoints)
            self.assertEqual(len(receipts),16)
            self.assertTrue(verify_common_stream_prefix(1,case,19,receipts))
            self.assertEqual([row["queries"] for row in rows],[0,7,16])
            self.assertEqual([sum(map(sum,row["count_by_cell"])) for row in rows],
                             [0,7,16])
            self.assertEqual([row["policy"] for row in rows],[policy]*3)
            self.assertTrue(all("hits_by_cell" in row for row in rows))
            self.assertEqual((rows,receipts),
                run_fractional_policy(1,case,19,policy,checkpoints))
        baseline,history=original_run_policy(1,case,19,"round_robin",checkpoints)
        self.assertTrue(verify_common_stream_prefix(1,case,19,history))
        self.assertEqual(len(baseline),3)

    def test_full_reporting_on_small_paired_pilot(self):
        scenarios=SCENARIOS[:2]
        seeds=SEEDS[:2]
        checkpoints=(0,8,16)
        out=build_pilot_report(scenarios=scenarios,seeds=seeds,checkpoints=checkpoints)
        self.assertEqual(out["run_counts"],{
            "trajectories":20,
            "snapshots":60,
            "queries":320,
            "paired_stream_verifications":20,
        })
        self.assertEqual(len(out["groups"]),30)
        self.assertEqual(len(out["paired_comparisons"]),24)
        self.assertEqual(len(out["trial_rows"]),60)
        self.assertEqual(out,build_pilot_report(
            scenarios=scenarios,seeds=seeds,checkpoints=checkpoints))
        self.assertEqual(set(out["policies"]),set(POLICIES))
        self.assertTrue(all(not r["false_cert"] or not r["true_rates_covered"]
                            for r in out["trial_rows"]))

    def test_signed_paired_deltas_independently_recomputed(self):
        out=build_pilot_report(
            scenarios=SCENARIOS[:1],seeds=SEEDS[:2],checkpoints=(0,12))
        for rec in out["paired_comparisons"]:
            rows=[r for r in out["trial_rows"] if
                  r["scenario"]==rec["scenario"] and r["queries"]==rec["budget"]]
            lookup={(r["seed"],r["policy"]):Q(r["true_frozen_error"])
                    for r in rows}
            delta=sum((lookup[(seed,rec["baseline"])]-
                       lookup[(seed,"fractional_cover")] for seed in SEEDS[:2]),
                      Q(0))/2
            self.assertEqual(Q(rec["paired_mean_true_error_improvement"]),delta)
        # The schema preserves all wins, losses and ties; negative means
        # the fixed fractional plan had higher measured error.

if __name__=="__main__":
    unittest.main()
