"""NOR-18: unit tests for exact work accounting and public certificate safety."""
from __future__ import annotations

from dataclasses import replace
import random
import unittest

from research.nor18_costed_prefix import experiment as n
from research.rrc_compactification import compound25 as corpus


class NOR18CostedPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck,p1,p2=n.nor11.train_source()
        cls.ck=ck
        rng=random.Random(2026111818)
        for i in range(100):
            case=corpus.make_case(rng,"late_source",f"nor18-unit-{i}")
            headers=corpus.possible_headers(case)
            if len(headers)==1:
                cls.case=case
                cls.header=headers[0]
                break
        else:
            raise AssertionError("source generator failed to provide unique header")

    def test_full_check_cost_matches_explicit_three_unary_and_composition(self):
        h=self.header
        n192=corpus.OBSERVED
        for j in range(16):
            p=n.nor17.program(h,j)
            want=2*(sum(n.unary_work(op,n192) for op in
                        (p[0],p[4],p[5]))
                    +n.composition_work(h,n192,True))
            self.assertEqual(n.full_check_work(h,j),want)
        self.assertEqual(n.unary_work("R",192),18528)
        self.assertEqual(n.unary_work("S",192),193)
        self.assertEqual(n.unary_work("N",192),193)
        with self.assertRaises(ValueError):
            n.unary_work("Z",192)

    def test_exact_cached_full_batch_unary_saving_identity(self):
        # This is a 4x4 cross-product of U/V operator choices.
        for header in (
            self.header,corpus.Header("I",64,24,1),
            corpus.Header("R",112,32,-1)):
            for depth in n.PREFIX_DEGREES+(192,):
                independent=sum(n.full_check_work(header,j,depth)
                                for j in range(16))
                cached=n.cached_screen_work(header,depth)
                ops=sum(n.unary_work(op,depth) for op in corpus.OPS)
                expected_saving=2*(16*n.unary_work(header.base,depth)+7*ops)
                self.assertEqual(independent-cached,expected_saving)
                self.assertGreater(independent,cached)
        with self.assertRaises(ValueError):
            n.cached_screen_work(self.header,96)

    def test_work_unit_is_public_header_determined(self):
        h=self.header
        a=n.cost_register(h)
        b=n.cost_register(h)
        self.assertEqual(a,b)
        self.assertEqual(a["public_full_check_unit"],
            (sum(a["full_check_costs"])+15)//16)
        self.assertGreater(a["query_simulation_cost"],0)
        self.assertEqual(set(a["screen_setup_costs"]),
                         {"112","128","144","160","176","192"})

    def test_actual_public_prefix_screen_is_sound_and_nested(self):
        h=self.header
        masks=[n.nor17.prefix_mask(self.case,h,d)
               for d in n.PREFIX_DEGREES+(192,)]
        self.assertEqual(len(masks),6)
        for coarse,fine in zip(masks,masks[1:]):
            self.assertEqual(coarse&fine,fine)
        full=0
        for j in range(16):
            if n.nor17.accepted_full(self.case,h,j):
                full|=1<<j
        self.assertEqual(full,masks[-1])
        self.assertGreater(full.bit_count(),0)

    def test_public_only_future_and_program_mutations_cannot_change_choices(self):
        p=n.policies_before_grading(self.case,self.header,self.ck)
        modified=replace(self.case,program=None,
            future=tuple(987654321 for _ in self.case.future))
        q=n.policies_before_grading(modified,self.header,self.ck)
        self.assertEqual(p,q)
        predictions,screens,register,original,arms,mask=p
        self.assertEqual(set(arms),set(n.ARM_NAMES))
        self.assertEqual(len(predictions),16)
        for j,prediction in enumerate(predictions):
            self.assertEqual(prediction,corpus.execute(
                self.case.query,n.nor17.program(self.header,j)))
        for name,z in arms.items():
            self.assertLessEqual(z["total_work"],z["work_cap"])
            self.assertEqual(z["total_work"],
                z["query_work"]+z["screen_work"]+z["full_candidate_check_work"])
            self.assertEqual(z["full_candidate_checks"],len(z["trace"]))
            if z["chosen_class"] is not None:
                self.assertTrue(mask&(1<<z["chosen_class"]))
            if name.startswith("greedy"):
                self.assertFalse(z["certified"])
                self.assertEqual(z["screen_work"],0)
                self.assertEqual(z["query_work"],0)
            if name.startswith("batch192") and z["setup_feasible"]:
                self.assertEqual(z["batched_degree192_public_comparisons"],16)
                self.assertEqual(z["full_candidate_checks"],0)
                self.assertEqual(z["screen_public_comparisons"],16)

    def test_batched_exact_full_mask_already_contains_accepted_witness(self):
        h=corpus.Header("I",64,24,1)
        work=n.cost_register(h)
        preds=[("A",)]*16
        full=(1<<0)|(1<<7)
        # The full public batch has ALREADY checked all sixteen candidates.
        def should_not_call(_):
            raise AssertionError("batch192 must not rebill separate full check")
        b=n.decide_budgeted(None,h,should_not_call,preds,list(range(16)),
            work,"batch",192,16,full)
        self.assertTrue(b["certified"])
        self.assertEqual(b["chosen_class"],0)
        self.assertEqual(b["full_candidate_checks"],0)
        self.assertEqual(b["batched_degree192_public_comparisons"],16)
        self.assertLessEqual(b["total_work"],b["work_cap"])

    def test_cost_budget_prevents_unpaid_public_verification(self):
        h=corpus.Header("I",64,24,1)
        work=n.cost_register(h)
        preds=[("A",)]*16
        attempted=[]
        def checked(j):
            attempted.append(j)
            return j==0
        result=n.decide_budgeted(None,h,checked,preds,list(range(16)),
            work,"consensus",176,4,(1<<16)-1)
        self.assertLessEqual(result["total_work"],result["work_cap"])
        self.assertEqual(len(attempted),result["full_candidate_checks"])
        self.assertEqual(len(attempted),len(result["trace"]))
        if result["certified"]:
            self.assertEqual(result["chosen_class"],0)
        # A setup that does not fit may only abstain, without charging work
        # that an inference policy did not get to run.
        tiny={**work,"query_simulation_cost":10**12}
        no=n.decide_budgeted(None,h,checked,preds,list(range(16)),
            tiny,"consensus",176,4,(1<<16)-1)
        self.assertEqual(no["exit"],"setup_over_work_cap")
        self.assertEqual(no["total_work"],0)
        self.assertFalse(no["certified"])

    def test_frozen_source_budgets(self):
        for kwargs in ({"replicates":5},{"late_per":49},
                       {"signed_per":13},{"rational_per":13}):
            with self.assertRaises(ValueError):
                n.fresh_cohorts({},**kwargs)
        self.assertEqual(len(n.ARM_NAMES),28)
        with self.assertRaises(ValueError):
            n.decide_budgeted(None,self.header,
                lambda j:True,[("A",)]*16,list(range(16)),
                n.cost_register(self.header),"greedy",None,5,65535)
        with self.assertRaises(ValueError):
            n.decide_budgeted(None,self.header,
                lambda j:True,[("A",)]*16,list(range(16)),
                n.cost_register(self.header),"batch",160,4,65535)


if __name__=="__main__":
    unittest.main()
