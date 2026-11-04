"""QO-11: exact nested-gap invariants and paired acquisition controls."""
import unittest
from fractions import Fraction as Q
from itertools import product

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate, robust_optimum
)
from research.ingression_dynamics_v1.expected_gap_probe import (
    gap_bounds, future_bounds, beta_planning_probability,
    exact_next_gap_scores, choose_expected_gap_probe, plateau_witness
)
from research.ingression_dynamics_v1.expected_gap_pilot import (
    SCENARIOS, POLICIES, SEEDS, CHECKPOINTS,
    gap_policy_runner, full_report
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    verify_common_stream_prefix,run_policy as old_run_policy
)


class GapMathTests(unittest.TestCase):
    def test_realized_interval_refinement_never_enlarges_gap(self):
        initial_options=(
            (Q(0),Q(1)),(Q(0),Q(1,2)),
            (Q(1,4),Q(3,4)),(Q(1,2),Q(1))
        )
        verified=0
        for origin in product(initial_options,repeat=3):
            old=tuple((entry,) for entry in origin)
            refined=[]
            for lo,hi in origin:
                mid=(lo+hi)/2
                refined.append((lo,mid))
            new=tuple((entry,) for entry in refined)
            for budget in (1,2,3):
                R,L,G=gap_bounds(old,budget)
                R2,L2,G2=gap_bounds(new,budget)
                self.assertLessEqual(R2,R)
                self.assertGreaterEqual(L2,L)
                self.assertLessEqual(G2,G)
                self.assertGreaterEqual(G2,0)
                # independent original QO-07 partition optimizer
                r0,l0,_,_=robust_optimum(old,budget)
                r1,l1,_,_=robust_optimum(new,budget)
                self.assertEqual((R,L),(r0,l0))
                self.assertEqual((R2,L2),(r1,l1))
                verified+=1
        self.assertEqual(verified,192)

    def test_exact_next_observation_matches_real_ledger_copy(self):
        original=ProbeLedger(2,2,Q(1,20),10)
        for i,a,hit in ((0,0,0),(1,1,1),(0,0,1),(1,0,0),
                        (1,1,1),(0,1,0),(0,0,0)):
            original.observe(i,a,hit)
        for state,action,outcome in product(range(2),range(2),(0,1)):
            hypothetical=future_bounds(original,state,action,outcome)
            independent=ProbeLedger(2,2,Q(1,20),10)
            for x in original.receipts:
                independent.observe(*x)
            try:
                independent.observe(state,action,outcome)
            except ValueError:
                self.assertIsNone(hypothetical)
            else:
                self.assertEqual(hypothetical,independent.snapshot())
                self.assertEqual(sum(independent.counts[i][a]
                    for i in range(2) for a in range(2)),8)

    def test_conditional_lookahead_expected_gap_is_exact(self):
        ledger=ProbeLedger(2,1,Q(1,20),10)
        for observation in [0]*27+[1]*5:
            ledger.observe(0,0,observation)
        for observation in [1]*20+[0]*12:
            ledger.observe(1,0,observation)
        s,calls=exact_next_gap_scores(ledger,1)
        self.assertEqual(calls,4)
        for cell,score in s.items():
            self.assertIsNotNone(score)
            i,a=cell
            p=beta_planning_probability(ledger,i,a)
            self.assertEqual(p,score["beta_predictive_hit"])
            self.assertEqual(score["expected_next_gap"],
                p*score["hit_gap"]+(1-p)*score["miss_gap"])
            self.assertEqual(score["current_gap"]-score["expected_next_gap"],
                             score["expected_reduction"])
            self.assertGreaterEqual(score["expected_reduction"],0)
            self.assertLessEqual(score["hit_gap"],score["current_gap"])
            self.assertLessEqual(score["miss_gap"],score["current_gap"])

    def test_true_lookahead_plateau_and_explicit_fallback(self):
        evidence=plateau_witness()
        self.assertEqual(evidence["starting_gap"],"1/2")
        self.assertEqual(evidence["all_one_step_gains"],("0","0"))
        ledger=ProbeLedger(2,1,Q(1,20),10)
        cell,receipt=choose_expected_gap_probe(ledger,1)
        self.assertEqual(cell,(0,0))
        self.assertEqual(receipt["mode"],"fallback_bottleneck")
        self.assertEqual(receipt["hypothetical_optimizer_calls"],4)
        cell2,receipt2=choose_expected_gap_probe(ledger,1,fallback="width")
        self.assertEqual(cell2,(0,0))
        self.assertEqual(receipt2["mode"],"fallback_width")

    def test_planning_prior_is_not_a_confidence_bound(self):
        ledger=ProbeLedger(1,1,Q(1,20),10)
        for hit in (1,1,0):
            ledger.observe(0,0,hit)
        self.assertEqual(beta_planning_probability(ledger,0,0),
                         Q(3,5))
        before=ledger.snapshot()
        for a,b in ((Q(1),Q(1)),(Q(2),Q(1))):
            _,meta=choose_expected_gap_probe(
                ledger,1,prior_success=a,prior_failure=b)
            self.assertEqual(meta["candidate_cells"],1)
            self.assertEqual(ledger.snapshot(),before)
        with self.assertRaises(ValueError):
            beta_planning_probability(ledger,0,0,Q(0),Q(1))
        with self.assertRaises(ValueError):
            choose_expected_gap_probe(ledger,1,fallback="bad")

    def test_validity_of_gap_for_observed_confidence_ledger(self):
        ledger=ProbeLedger(3,2,Q(1,20),10)
        old=gap_bounds(ledger.snapshot(),1)
        for i,a,hit in ((0,0,0),(1,1,1),(1,0,1),
                        (2,0,0),(2,1,1),(1,1,0)):
            ledger.observe(i,a,hit)
            new=gap_bounds(ledger.snapshot(),1)
            self.assertLessEqual(new[0],old[0])
            self.assertGreaterEqual(new[1],old[1])
            self.assertLessEqual(new[2],old[2])
            old=new


class GapPilotTests(unittest.TestCase):
    def test_source_fixed_cases_and_prospective_labeling(self):
        self.assertEqual(len(SCENARIOS),5)
        self.assertEqual(SEEDS,(401,409,419,421))
        self.assertEqual(CHECKPOINTS,(32,96,192))
        self.assertEqual(POLICIES,(
            "round_robin","width","bottleneck",
            "expected_gap_bottleneck","expected_gap_width"))
        self.assertTrue(all("panel" in x for x in SCENARIOS))
        self.assertEqual(sum("new source-frozen" in x["panel"] for x in SCENARIOS),2)

    def test_replay_and_equal_query_budget(self):
        c=SCENARIOS[0]
        for policy in ("expected_gap_bottleneck","expected_gap_width"):
            a,receipts,cost=gap_policy_runner(0,c,401,policy,(0,6,12))
            b,replay,recomputed=gap_policy_runner(0,c,401,policy,(0,6,12))
            self.assertEqual((a,receipts,cost),(b,replay,recomputed))
            self.assertEqual(len(receipts),12)
            self.assertTrue(verify_common_stream_prefix(0,c,401,receipts))
            self.assertEqual(cost["plateau_steps"]+cost["positive_steps"],12)
            self.assertTrue(cost["gap_nonincreasing"])
            self.assertEqual([x["queries"] for x in a],[0,6,12])
            self.assertTrue(all(not x["false_cert"] or
                                not x["true_rates_covered"] for x in a))
        base,observed=old_run_policy(0,c,401,"bottleneck",(0,6,12))
        self.assertTrue(verify_common_stream_prefix(0,c,401,observed))

    def test_full_small_panel_and_paired_deltas(self):
        c=SCENARIOS[:2]
        seeds=SEEDS[:2]
        checkpoints=(0,6,12)
        report=full_report(scenarios=c,seeds=seeds,checkpoints=checkpoints)
        self.assertEqual(report["run_counts"],{
            "trajectories":20,"snapshots":60,
            "queries":240,"stream_replays":20,"planning_traces":8
        })
        self.assertEqual(len(report["groups"]),30)
        self.assertEqual(len(report["paired_comparisons"]),48)
        self.assertEqual(len(report["trial_rows"]),60)
        self.assertEqual(len(report["trajectory_receipt_hashes"]),20)
        self.assertEqual(report,full_report(
            scenarios=c,seeds=seeds,checkpoints=checkpoints))
        for row in report["trial_rows"]:
            self.assertGreaterEqual(Q(row["certification_gap"]),0)
        for delta in report["paired_comparisons"]:
            relevant=[r for r in report["trial_rows"]
                      if r["scenario"]==delta["scenario"]
                      and r["queries"]==delta["budget"]]
            lookup={(r["seed"],r["policy"]):Q(r["certification_gap"])
                    for r in relevant}
            brute=sum((
                lookup[s,delta["baseline"]]-lookup[s,delta["candidate"]]
                for s in seeds),Q(0))/len(seeds)
            self.assertEqual(Q(delta["mean_gap_improvement"]),brute)

    def test_invalid_pilot_inputs(self):
        with self.assertRaises(ValueError):
            gap_policy_runner(0,SCENARIOS[0],401,"invalid",(0,4))
        with self.assertRaises(ValueError):
            gap_policy_runner(0,SCENARIOS[0],401,
                              "expected_gap_width",(4,4))
        with self.assertRaises(ValueError):
            full_report(scenarios=SCENARIOS[:1],seeds=(401,401),
                        checkpoints=(0,4))


if __name__=="__main__":
    unittest.main()
