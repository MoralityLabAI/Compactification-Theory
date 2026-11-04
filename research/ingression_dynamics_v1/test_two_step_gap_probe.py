"""QO-12 exact two-stage Bellman oracle and paired cost-accounting tests."""
import unittest
from fractions import Fraction as Q
from itertools import product

from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger
from research.ingression_dynamics_v1.expected_gap_probe import (
    gap_bounds, exact_next_gap_scores
)
from research.ingression_dynamics_v1.two_step_gap_probe import (
    planning_state, optimal_horizon_value, choose_two_step_on_plateau,
    _child, two_cell_exact_plateau
)
from research.ingression_dynamics_v1.two_step_gap_pilot import (
    SCENARIOS,SEEDS,POLICIES,CHECKPOINTS,run_two_step,complete_report
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy,verify_common_stream_prefix
)


def independent_replayed_ledger(start, steps):
    ledger=ProbeLedger(start.n,start.d,start.alpha,start.bits)
    for i,a,y in tuple(start.receipts)+tuple(steps):
        ledger.observe(i,a,y)
    return ledger


def independent_two_step_value(ledger, classes):
    """Independent exhaustive policy enumeration via actual QO-07 ledger replay.

    Evaluate each first cell and, separately after each observed first
    outcome, the optimal second cell. No reuse of QO-12's state/update
    representation or memoized optimizer.
    """
    cells=tuple(product(range(ledger.n),range(ledger.d)))
    def probability(state,i,a):
        return Q(1+state.hits[i][a],2+state.counts[i][a])
    best=None
    for c in cells:
        p=probability(ledger,*c)
        values=[]
        for y in (0,1):
            try:
                first=independent_replayed_ledger(ledger,((c[0],c[1],y),))
            except ValueError:
                values.append(None)
                continue
            options=[]
            for d in cells:
                q=probability(first,*d)
                future=[]
                for z in (0,1):
                    try:
                        child=independent_replayed_ledger(first,((d[0],d[1],z),))
                    except ValueError:
                        future.append(None)
                    else:
                        future.append(gap_bounds(child.snapshot(),classes)[2])
                if all(g is not None for g in future):
                    options.append((1-q)*future[0]+q*future[1])
            values.append(min(options) if options else None)
        if all(v is not None for v in values):
            candidate=(1-p)*values[0]+p*values[1]
            if best is None or candidate<best:
                best=candidate
    return best


class TwoStepMathTests(unittest.TestCase):
    def test_exact_plateau_and_positive_two_step_acquisition(self):
        witness=two_cell_exact_plateau()
        self.assertEqual(witness["unrounded_expected_two_step_reduction"],"1/320")
        ledger=ProbeLedger(2,1,Q(1,20),10)
        scores,calls=exact_next_gap_scores(ledger,1)
        self.assertEqual(calls,4)
        self.assertTrue(all(s["expected_reduction"]==0 for s in scores.values()))
        one=optimal_horizon_value(ledger,1,1)
        two=optimal_horizon_value(ledger,1,2)
        self.assertEqual(one["expected_reduction"],0)
        self.assertGreater(two["expected_reduction"],0)
        self.assertLessEqual(two["expected_reduction"],Q(1,320))
        self.assertEqual(two["first_cell"],(0,0))
        self.assertEqual(two["second_if_hold"]["first_cell"],(1,0))
        self.assertEqual(two["second_if_hit"]["first_cell"],(1,0))
        self.assertEqual(two["expected_remaining_gap"],
                         independent_two_step_value(ledger,1))
        cell,audit=choose_two_step_on_plateau(ledger,1)
        self.assertEqual(cell,(0,0))
        self.assertEqual(audit["mode"],"two_step_positive")
        self.assertGreater(audit["two_step_gap_calls"],0)

    def test_counterfactual_child_replays_real_ledger(self):
        ledger=ProbeLedger(2,2,Q(1,20),10)
        for rec in ((0,0,1),(1,0,0),(0,1,0),(1,1,1),(1,0,1)):
            ledger.observe(*rec)
        state=planning_state(ledger)
        for c in product(range(2),range(2)):
            for y in (0,1):
                hypothetical=_child(state,c,y,ledger.alpha,ledger.bits)
                try:
                    replay=independent_replayed_ledger(ledger,((c[0],c[1],y),))
                except ValueError:
                    self.assertIsNone(hypothetical)
                else:
                    self.assertEqual(hypothetical,planning_state(replay))

    def test_optimizing_second_probe_can_depend_on_first_outcome(self):
        ledger=ProbeLedger(2,1,Q(1,20),10)
        for y in (1,0,1,0,1,1,0,1):
            ledger.observe(0,0,y)
        for y in (0,0,1,0,0,1,0,0):
            ledger.observe(1,0,y)
        scores=[optimal_horizon_value(ledger,1,h) for h in (0,1,2)]
        self.assertGreaterEqual(scores[0]["expected_remaining_gap"],
                                scores[1]["expected_remaining_gap"])
        self.assertGreaterEqual(scores[1]["expected_remaining_gap"],
                                scores[2]["expected_remaining_gap"])
        self.assertEqual(scores[2]["expected_remaining_gap"],
                         independent_two_step_value(ledger,1))

    def test_all_three_stage_scores_nested_for_small_real_ledger(self):
        for n in (2,3):
            ledger=ProbeLedger(n,1,Q(1,20),10)
            for i in range(n):
                for j in range(4):
                    ledger.observe(i,0,(i+j)%2)
            v=[optimal_horizon_value(ledger,1,h) for h in (0,1,2)]
            self.assertTrue(all(x["expected_remaining_gap"] is not None for x in v))
            self.assertGreaterEqual(v[0]["expected_remaining_gap"],
                                    v[1]["expected_remaining_gap"])
            self.assertGreaterEqual(v[1]["expected_remaining_gap"],
                                    v[2]["expected_remaining_gap"])

    def test_prior_controls_only_planning(self):
        ledger=ProbeLedger(2,1,Q(1,20),10)
        old=ledger.snapshot()
        before=optimal_horizon_value(ledger,1,2,Q(1),Q(1))
        changed=optimal_horizon_value(ledger,1,2,Q(2),Q(1))
        self.assertEqual(ledger.snapshot(),old)
        self.assertGreaterEqual(before["expected_reduction"],0)
        self.assertGreaterEqual(changed["expected_reduction"],0)
        with self.assertRaises(ValueError):
            optimal_horizon_value(ledger,1,2,Q(0),Q(1))
        with self.assertRaises(ValueError):
            optimal_horizon_value(ledger,1,3)
        with self.assertRaises(ValueError):
            choose_two_step_on_plateau(ledger,1,fallback="unknown")


class TwoStepPilotTests(unittest.TestCase):
    def test_registered_casebook_and_ledger_protocol(self):
        self.assertEqual(POLICIES,(
            "bottleneck","expected_gap_bottleneck",
            "two_step_bottleneck","two_step_width"))
        self.assertEqual(SEEDS,(449,457,461))
        self.assertEqual(CHECKPOINTS,(24,64,96))
        self.assertEqual(len(SCENARIOS),5)
        self.assertTrue(all("panel" in case for case in SCENARIOS))
        self.assertEqual(sum("new source-frozen" in c["panel"] for c in SCENARIOS),2)

    def test_twelve_queries_paired_and_cost_complete(self):
        case=SCENARIOS[0]
        for policy in ("two_step_bottleneck","two_step_width"):
            a,receipts,cost=run_two_step(0,case,449,policy,(0,4,12))
            b,replay,cost2=run_two_step(0,case,449,policy,(0,4,12))
            self.assertEqual((a,receipts,cost),(b,replay,cost2))
            self.assertEqual(len(receipts),12)
            self.assertEqual([r["queries"] for r in a],[0,4,12])
            self.assertTrue(verify_common_stream_prefix(0,case,449,receipts))
            self.assertEqual(sum(cost[k] for k in (
                "one_step_positive","two_step_positive",
                "fallback_bottleneck","fallback_width")),12)
            self.assertTrue(all(not x["false_cert"] or not x["true_rates_covered"]
                                for x in a))
        rows,events=run_policy(0,case,449,"bottleneck",(0,4,12))
        self.assertTrue(verify_common_stream_prefix(0,case,449,events))

    def test_full_small_paired_report_with_negative_results_preserved(self):
        cases=SCENARIOS[:2]
        seeds=SEEDS[:2]
        budgets=(0,4,12)
        report=complete_report(scenarios=cases,seeds=seeds,checkpoints=budgets)
        self.assertEqual(report["run_counts"],{
            "trajectories":16,"snapshots":48,"queries":192,
            "stream_replays":16,"two_step_planning_traces":8})
        self.assertEqual(len(report["groups"]),24)
        self.assertEqual(len(report["paired_comparisons"]),36)
        self.assertEqual(len(report["trajectory_hashes"]),16)
        self.assertEqual(len(report["planning_receipts"]),8)
        self.assertEqual(report,complete_report(
            scenarios=cases,seeds=seeds,checkpoints=budgets))
        for row in report["trial_rows"]:
            self.assertGreaterEqual(Q(row["certification_gap"]),0)
            self.assertTrue(not row["false_cert"] or not row["true_rates_covered"])
        for p in report["paired_comparisons"]:
            chosen=[r for r in report["trial_rows"]
                    if r["scenario"]==p["scenario"]
                    and r["queries"]==p["budget"]]
            by={(r["seed"],r["policy"]):Q(r["certification_gap"])
                for r in chosen}
            independent=sum((
                by[s,p["baseline"]]-by[s,p["candidate"]]
                for s in seeds),Q(0))/len(seeds)
            self.assertEqual(Q(p["paired_mean_gap_improvement"]),independent)


if __name__=="__main__":
    unittest.main()
