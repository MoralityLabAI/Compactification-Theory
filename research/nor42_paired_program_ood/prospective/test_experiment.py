"""NOR-42 prospective pre-data checks on completed NOR41 sources only.

No test invokes NOR42 new_source/make_fixture/restore_worlds/evaluate, or
samples any future NOR42 source identity, suffix beta/transform or EVAL grade.
"""
from __future__ import annotations
import json
import random
import unittest
from fractions import Fraction
from itertools import combinations
from math import comb
from zipfile import ZipFile

from research.nor42_paired_program_ood.prospective import experiment as x


class NOR42PreData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with ZipFile(x.ROOT/"research/nor41_scale_ood/results/nor41_original_first_hosted.zip") as z:
            fixture=json.loads(z.read("nor41_fixture.json"))
        row=fixture["cases"][0]
        assert row["replicate"]==84 and row["degree"]==512
        cls.row=row
        cls.pred=x.nor37.public_proposal(row)
        cls.src=row["full_public_input_sha256"]

    def test_registration_and_exact_historical_388_id_sha(self):
        self.assertEqual(x.REGISTERED,"9bf353b7c76535e581ff026098cd682914a90ef0")
        self.assertEqual(x.REPS,tuple(range(100,116)))
        self.assertEqual(x.FAMILIES,("wide_dense_small","wide_regime_blocks"))
        self.assertEqual(x.OFFSET,130042000)
        self.assertEqual(x.K,16)
        self.assertEqual(x.EVAL,1024)
        self.assertEqual(x.S_SUFFIX,202611420091)
        self.assertEqual(len(x.history()),388)
        self.assertIn(self.src,x.history())
        self.assertEqual(len(x.POLICIES),4)
        self.assertEqual(len(x.ARMS),6)

    def test_only_archived_nor41_public_fixture(self):
        row=self.row
        pred=self.pred["pred"]
        actual=x.nor41.restore_native_target(row,pred)
        self.assertEqual(pred,actual)
        self.assertGreater(row["original_native_public_mask"],0)
        self.assertNotIn("program",row)
        self.assertNotIn("future",row)
        self.assertNotIn("target",row)
        self.assertEqual(len(pred),769)

    def test_hidden_suffix_prior_worlds_identical_public_prefix(self):
        row=self.row
        pred=self.pred["pred"]
        query=tuple(row["public_query_input"])
        d=row["degree"]
        for beta in x.BETAS:
            for op in x.corpus.OPS:
                hidden=x.hidden_future(query,pred,d,beta,op)
                self.assertEqual(hidden[:d+1],pred[:d+1])
                self.assertEqual(hidden[:d+32],pred[:d+32])
                self.assertEqual(hidden[d+32]-pred[d+32],beta)
                changed=[i for i,(a,b) in enumerate(zip(pred,hidden)) if a!=b]
                self.assertEqual(min(changed),d+32)
                self.assertTrue(changed)

    def test_exact_uniform_probability_on_small_combinatorial_game(self):
        # The prospective uniform law uses K=16, enumerate m=20 supports.
        m=20
        for diff in (1,2,3,4,8,16,20):
            S=set(range(diff))
            hit=sum(bool(S.intersection(a)) for a in combinations(range(m),x.K))
            total=comb(m,x.K)
            self.assertEqual(x.uniform_exact(m,diff),Fraction(hit,total))

    def test_policies_and_heldout_plan_quotas_using_archived_source(self):
        old,cats,u,cu,cv,bu,bv,scores=x.nor37.source_geometry(
            self.pred["pred"],self.pred["header"],self.src)
        plans=x.schedules(old,cats,scores,self.src)
        self.assertEqual(set(plans),set(x.MODES))
        self.assertEqual(len(plans["original"]["uniform"]),1024)
        self.assertNotEqual(x.eval_tag(self.src),x.nor41.phase_tag(self.src,"eval"))
        for p in (*x.POLICIES,x.DIAGNOSTIC):
            self.assertEqual(sum(w for _,_,w in x.components(p)),256)
        self.assertEqual(x.components("global7_mix")[0],
                         ("original","uniform",112))
        for arm in x.ARMS:
            for t in (0,11,1023):
                base=set(plans["original"][arm][t])
                self.assertEqual(len(base),16)
                quotas=[len(base&set(cat)) for cat in cats]
                for mode in ("atom_curvature50","atom_random"):
                    other=set(plans[mode][arm][t])
                    self.assertEqual(len(other),16)
                    self.assertEqual([len(other&set(cat)) for cat in cats],quotas)

    def test_synthetic_hidden_oracle_versus_public_self_consistency(self):
        clean=self.pred["pred"]
        d=self.row["degree"]
        hidden=x.hidden_future(tuple(self.row["public_query_input"]),clean,d,1,"I")
        visible=tuple(range(16))
        hits=(d+32,)+tuple(range(15))
        self.assertTrue(x.nor24.exact_witness_accepts(
            clean,hidden,visible,frozenset(),1))
        self.assertFalse(x.nor24.exact_witness_accepts(
            clean,hidden,hits,frozenset(),1))
        self.assertTrue(x.nor24.exact_witness_accepts(
            clean,clean,hits,frozenset(),1))
        self.assertTrue(x.nor24.exact_witness_accepts(
            clean,clean,visible,frozenset(),4294049777))

    def test_frozen_five_science_gates_on_synthetic_records(self):
        rows=[];recs=[]
        for rep in x.REPS:
            for d in x.DEGREES:
                for family in x.FAMILIES:
                    rows.append({
                        "replicate":rep,"degree":d,"family":family,
                        "absolute_uniform_calibration_error":.005,
                        "global7_detection":.75,"uniform_detection":.80,
                        "global7_minus_uniform":-.05,
                        "global7_minus_exact_floor":.25
                    })
                    recs.append({
                        "public_sha256":f"synthetic-{rep}-{d}-{family}",
                        "paired_public_demos_equal":True,
                        "same_public_query_input":True,
                        "native_public_mask_same_worlds":True,
                        "hidden_outside_original_admissible_grammar":True,
                        "public_model_exactly_one_of_two_worlds":True,
                        "first_changed_index":d+32,
                        "degree":d,"first_hidden_coefficient_delta":1,
                        "hidden_beta":1,"difference_support_size":42,
                        "authentic_clean_false_rejections":0,
                        "two_delta_direct_witness_disagreements":0,
                        "direct_integer_witness_checks":6,
                        "witnessfree_self_consistency_accepts_both_worlds":True,
                        "paired_worlds":2,"public_predictor_correct_worlds":1,
                        "non16_or_duplicate_component_plans":0,
                        "policy_mixture_weight_mismatch":0,
                        "eight_atom_quota_mismatches":0,
                        "mixture_detection_numerators":{p:1 for p in (*x.POLICIES,x.DIAGNOSTIC)}
                    })
        means,wins,gates=x.primary_gates(recs,rows)
        self.assertTrue(all(gates.values()))
        self.assertEqual(len(rows),64)
        self.assertEqual(wins["ood_global7_within_15pp_uniform"],32)
        for row in rows:
            if row["family"]=="wide_regime_blocks":
                row["global7_minus_uniform"]=-.30
        _,_,altered=x.primary_gates(recs,rows)
        self.assertFalse(altered["global7_competitive_offgrammar_new_family"])


if __name__=="__main__":
    unittest.main()
