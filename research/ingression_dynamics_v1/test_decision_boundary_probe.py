"""QO-10 fixed-boundary LP, paired-stream and complete-report regressions."""
import unittest
from fractions import Fraction as Q
from itertools import product

from research.ingression_dynamics_v1.decision_boundary_probe import (
    SCENARIOS, SEEDS, CHECKPOINTS, POLICIES, BASE_POLICIES,
    TRAIN_BASE, TRAIN_CHANGED, TOLERANCE,
    declared_alternative_masks, make_training_world,
    decision_boundary_plan, run_decision_policy, full_report,
)
from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as baseline_run_policy,verify_common_stream_prefix
)
from research.ingression_dynamics_v1.fractional_cover_probe import (
    run_fractional_policy, choose_fractional_cell
)


class QO10Tests(unittest.TestCase):
    def test_all_declared_training_worlds_cross_real_decision_boundary(self):
        self.assertEqual(optimum_for_k(make_training_world(0),1)[0],0)
        self.assertEqual(len(declared_alternative_masks()),6)
        for rotated in (False,True):
            plan=decision_boundary_plan(rotated)
            self.assertEqual(plan["exact_lp_objective"],Q(9,2))
            self.assertEqual(plan["baseline_error"],Q(0))
            self.assertEqual(tuple(a for a,b in plan["alternative_errors"]),
                             plan["alternative_masks"])
            self.assertTrue(all(Q(err)==Q(2,5) for mask,err
                                in plan["alternative_errors"]))
            self.assertTrue(all(optimum_for_k(make_training_world(mask),1)[0]
                                >TOLERANCE for mask in plan["alternative_masks"]))
            self.assertEqual(sum(plan["normalized_shares"],Q(0)),Q(1))

    def test_exact_independent_half_grid_lp_and_rotated_certificate(self):
        for rotated,expected in (
            (False,(Q(1),Q(1,2),Q(1),Q(1,2),Q(1),Q(1,2))),
            (True,(Q(1,2),Q(1),Q(1,2),Q(1),Q(1,2),Q(1)))):
            plan=decision_boundary_plan(rotated)
            self.assertEqual(plan["primal_weights"],expected)
            self.assertEqual(sum(plan["dual_weights"],Q(0)),Q(9,2))
            masks=plan["alternative_masks"]
            independent=min(
                sum(x,Q(0)) for x in product((Q(0),Q(1,2),Q(1)),repeat=6)
                if all(sum((x[i] for i in range(6) if mask&(1<<i)),Q(0))
                       >=1 for mask in masks))
            self.assertEqual(independent,Q(9,2))

    def test_casebook_and_out_of_family_negative_control(self):
        self.assertEqual(len(SCENARIOS),5)
        self.assertEqual(SEEDS,(101,103,107,109,113,127))
        self.assertEqual(CHECKPOINTS,(32,96,192))
        self.assertEqual(POLICIES,(
            "round_robin","width","bottleneck","fractional_cover",
            "decision_boundary","decision_boundary_rotated"))
        true_labels=[]
        for case in SCENARIOS:
            self.assertEqual(len(case["rates"]),3)
            self.assertTrue(all(len(row)==2 for row in case["rates"]))
            expected=max(
                (max(row[a] for row in case["rates"])-
                 min(row[a] for row in case["rates"]))/2
                for a in range(2))
            actual=optimum_for_k(case["rates"],case["classes"])[0]
            self.assertEqual(actual,expected)
            true_labels.append(actual<=case["tolerance"])
        self.assertEqual(true_labels,[True,False,False,False,True])
        # The action-one singleton is a genuine alternative to feasible,
        # while the registered training family contains only action-one pairs.
        action1_singleton=1<<3
        self.assertNotIn(action1_singleton,declared_alternative_masks())
        self.assertEqual(optimum_for_k(make_training_world(
            action1_singleton),1)[0],Q(2,5))

    def test_no_test_rate_peeking_in_static_first_action(self):
        plan=decision_boundary_plan()
        other=decision_boundary_plan(True)
        c=[[0,0] for _ in range(3)]
        first=choose_fractional_cell(c,plan)
        rotated=choose_fractional_cell(c,other)
        self.assertEqual(first,(0,0))
        self.assertEqual(rotated,(0,0))
        for i,a in product(range(3),range(2)):
            c[i][a]+=1
        self.assertEqual(choose_fractional_cell(c,plan),
                         choose_fractional_cell([[1,1] for _ in range(3)],plan))
        # Every cell has positive declared weight, so no cell is unsampled.
        for rotated in (False,True):
            plan=decision_boundary_plan(rotated)
            counts=[[0,0] for _ in range(3)]
            first_six=[]
            for t in range(192):
                i,a=choose_fractional_cell(counts,plan)
                if t<6: first_six.append((i,a))
                counts[i][a]+=1
            self.assertEqual(set(first_six),set(product(range(3),range(2))))
            self.assertTrue(all(x>0 for row in counts for x in row))
            self.assertEqual(sum(map(sum,counts)),192)

    def test_paired_prefix_and_budget_against_unchanged_baselines(self):
        case=SCENARIOS[1]
        snapshots=(0,8,16)
        for policy in POLICIES:
            if policy in BASE_POLICIES:
                rows,receipts=baseline_run_policy(1,case,101,policy,snapshots)
            elif policy=="fractional_cover":
                rows,receipts=run_fractional_policy(1,case,101,policy,snapshots)
            else:
                rows,receipts=run_decision_policy(1,case,101,policy,snapshots)
            self.assertEqual(len(receipts),16)
            self.assertTrue(verify_common_stream_prefix(1,case,101,receipts))
            self.assertEqual([x["queries"] for x in rows],[0,8,16])
            self.assertEqual([sum(map(sum,x["count_by_cell"])) for x in rows],
                             [0,8,16])
            self.assertTrue(all(not x["false_cert"] or
                                not x["true_rates_covered"] for x in rows))

    def test_complete_small_paired_report_and_deltas(self):
        cases=SCENARIOS[:2]
        seeds=SEEDS[:2]
        budgets=(0,8,16)
        result=full_report(scenarios=cases,seeds=seeds,checkpoints=budgets)
        self.assertEqual(result["run_counts"],{
            "trajectories":24,
            "snapshots":72,
            "queries":384,
            "stream_replays":24,
        })
        self.assertEqual(len(result["groups"]),36)
        self.assertEqual(len(result["paired_comparisons"]),30)
        self.assertEqual(len(result["trajectory_receipt_hashes"]),24)
        self.assertEqual(result,full_report(
            scenarios=cases,seeds=seeds,checkpoints=budgets))
        for entry in result["paired_comparisons"]:
            matched=[r for r in result["trial_rows"]
                     if r["scenario"]==entry["scenario"]
                     and r["queries"]==entry["budget"]]
            by={(r["seed"],r["policy"]):Q(r["true_frozen_error"])
                for r in matched}
            expected=sum((
                by[seed,entry["baseline"]]-by[seed,"decision_boundary"]
                for seed in seeds),Q(0))/len(seeds)
            self.assertEqual(Q(entry["mean_true_error_improvement"]),expected)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): make_training_world(64)
        with self.assertRaises(ValueError):
            run_decision_policy(0,SCENARIOS[0],101,"invalid",(0,8))
        with self.assertRaises(ValueError):
            run_decision_policy(0,SCENARIOS[0],101,"decision_boundary",(8,8))
        with self.assertRaises(ValueError):
            run_decision_policy(0,{**SCENARIOS[0],"classes":2},101,
                                "decision_boundary",(0,8))
        with self.assertRaises(ValueError):
            full_report(scenarios=SCENARIOS[:1],seeds=(101,101),
                        checkpoints=(0,8))


if __name__=="__main__":
    unittest.main()
