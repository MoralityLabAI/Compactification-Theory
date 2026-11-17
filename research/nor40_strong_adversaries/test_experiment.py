"""NOR-40 pre-data tests. Only archived NOR-39 source identities and synthetic masks."""
from __future__ import annotations
from itertools import combinations
from fractions import Fraction
from math import comb
from zipfile import ZipFile
from hashlib import sha256
import json,random,unittest

from research.nor40_strong_adversaries import experiment as x


class NOR40PreDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive=x.ROOT/"research/nor39_clean_global_uniform/results/nor39_original_first_hosted.zip"
        with ZipFile(archive) as z:
            cls.row=json.loads(z.read("nor39_fixture.json"))["cases"][0]
        assert cls.row["replicate"]==68
        cls.p=x.nor37.public_proposal(cls.row)
        cls.src=cls.row["full_public_input_sha256"]
        cls.old,cls.atoms,cls.u,cls.cu,cls.cv,cls.bu,cls.bv,cls.scores=(
            x.nor37.source_geometry(cls.p["pred"],cls.p["header"],cls.src))

    def test_immutable_new_cohort_registration_and_historical_roster(self):
        self.assertEqual(x.REGISTERED,
                         "dbc1adbe0e30b9c0e36421cf07cebcf376d6a1ec")
        self.assertEqual(x.REPS,tuple(range(76,84)))
        self.assertEqual(x.OFFSET,130040000)
        self.assertEqual(len(x.history()),276)
        self.assertIn(self.src,x.history())
        self.assertEqual((x.CAL,x.BASE_CAL,x.EVAL,x.ATTACKS,x.GREEDY_STARTS),
                         (2048,256,512,8,32))
        self.assertEqual((x.POOL_SIZE,x.SWAP_PASSES),(16,4))
        self.assertEqual(len(x.POLICIES),7)
        self.assertEqual(len(x.CONTROLS),6)
        self.assertEqual(sum(x.WEIGHTS),16)

    def test_archived_source_public_only(self):
        self.assertTrue(all(v not in self.row for v in (
            "program","future","target","attack","witness","audit")))
        self.assertEqual(self.row["original_native_public_mask"],self.p["mask"])
        self.assertEqual(x.nor39.restore_native_target(self.row,self.p["pred"]),
                         self.p["pred"])
        self.assertEqual(len(self.p["pred"]),769)

    def test_blind_genuine_hidden_eval_independence(self):
        self.assertNotEqual(x.phase_tag(self.src,"cal"),
                            x.phase_tag(self.src,"eval"))
        self.assertEqual(x.components("global7_mix")[0],
                         ("original","uniform",112))
        self.assertEqual(sum(w for _,_,w in x.components("global7_mix")),256)
        self.assertEqual(x.components("global8_mix")[0],
                         ("original","uniform",128))
        for m in (769,1537):
            uniform=Fraction(1)-Fraction(comb(m-16,4),comb(m,4))
            self.assertEqual(uniform,x.nor35.ideal_uniform_attack_detect(m))
            self.assertGreater(Fraction(7,16)*uniform,0)

    def fake_exposure(self,m=32):
        """Repeatable 256/2048-bit trial-set perturbations, no source program."""
        rng=random.Random(12345)
        bits={}
        for mode in x.MODES:
            for arm in x.ARMS:
                bits[(mode,arm)]=[
                    sum(1<<t for t in range(0,256,16)
                        if rng.randrange(3)==0) |
                    (rng.getrandbits(256)<<256)
                    for _ in range(m)]
        return bits

    def test_true_global_minimum_bracket_on_small_exhaustive_game(self):
        # As r=4 and m=7 the exact global optimum can be checked directly.
        cov=self.fake_exposure(7)
        for policy in x.POLICIES:
            single=x.singletons(cov,policy)
            globalmin=min(x.cal_objective(cov,policy,a)
                          for a in combinations(range(7),4))
            lb=sorted(single)[3]
            self.assertLessEqual(lb,globalmin)
            attack=x.greedy_attack(cov,policy,"a"*64,0)
            self.assertGreaterEqual(x.cal_objective(cov,policy,attack),
                                    globalmin)

    def test_2048_cal_search_exact_pool_optimality(self):
        cov=self.fake_exposure(28)
        old={(mode,arm):[mask & ((1<<256)-1) for mask in arr]
             for (mode,arm),arr in cov.items()}
        for policy in x.POLICIES:
            baseline,strong,cert=x.stronger_calibrated_attacks(
                cov,old,policy,"e"*64)
            self.assertEqual(len(baseline),len(strong))
            self.assertEqual(len(strong),8)
            self.assertEqual(cert["restricted_sets_enumerated"],1820)
            self.assertEqual(len(cert["full_cal_greedy32_sets"]),32)
            self.assertTrue(all(len(set(a))==4 for a in baseline+strong))
            pool=cert["restricted_candidate_pool"]
            best=min((x.cal_objective(cov,policy,a),a)
                     for a in combinations(pool,4))
            self.assertEqual((cert["restricted_exact_value"],
                              tuple(cert["restricted_exact_optimum"])),best)
            self.assertLessEqual(cert["global_singleton_lower_bound"],best[0])
            self.assertLessEqual(cert["global_best_found_upper_bound"],best[0])
            self.assertLessEqual(cert["strong_eight_full_cal_sum"],
                                 cert["baseline_eight_full_cal_sum"])
            self.assertEqual(strong,
                x.stronger_calibrated_attacks(cov,old,policy,"e"*64)[1])

    def test_historical_blind_attack_quota_and_zero_union(self):
        m=len(self.p["pred"])
        bits={(mode,arm):[1<<(j%17) for j in range(m)]
              for mode in x.MODES for arm in x.ARMS}
        attacks=x.blind_attacks(self.src,self.atoms,self.u,self.cu,self.cv,
                                self.bu,self.bv,m,bits)
        self.assertEqual(set(attacks),set(x.CONTROLS))
        self.assertTrue(all(len(sets)==8 for sets in attacks.values()))
        for a,b in zip(attacks["top_curvature"],
                       attacks["eight_atom_matched"]):
            self.assertEqual([sum(i in atom for i in a) for atom in self.atoms],
                             [sum(i in atom for i in b) for atom in self.atoms])
        self.assertTrue(all(len(set(a))==4 for a in attacks["zero_quota_stress"]))

    def test_registered_source_level_gates_from_synthetic_receipts(self):
        good_cert={p:{
            "restricted_sets_enumerated":1820,
            "global_singleton_lower_bound":10,
            "restricted_exact_value":20,
            "global_best_found_upper_bound":15,
            "global_optimality_gap_upper_minus_lower":5,
            "restricted_candidate_pool":list(range(16)),
            "strong_eight_full_cal_sum":120,
            "baseline_eight_full_cal_sum":160} for p in x.POLICIES}
        records=[{
            "public_sha256":str(i),"attack_certificates":good_cert,
            "native_mask":1,"public_prediction_sha256":"h",
            "private_native_truth_sha256_after_freeze":"h",
            "honest_target_false_rejections":0,
            "fully_schedule_informed_attacker_detections":0,
            "delta_decision_disagreements":0,"direct_authentic_witness_checks":10,
            "exact_quota_and_control_checks":{"quota":0},
            "exact_full_uniform_attack_detection_fraction":[1,10],
            "global7_exact_theoretical_floor":[1,20]
        } for i in range(48)]
        rows=[{
            "global7_baseline_minus_strong_selfplay":.004,
            "global7_strong_self_minus_deterministic_strong":.02,
            "global7_strong_worst_minus_original_strong_worst":-.003,
            "global7_strong_worst":.027,
            "global7_top_minus_atom_random":.04,
            "deterministic_top_minus_atom_random":.17
        } for _ in range(48)]
        _,wins,ratio,gates=x.primary_gates(records,rows)
        self.assertTrue(all(gates.values()))
        self.assertEqual(wins["strong_global7_beats_baseline_eval"],48)
        self.assertGreater(ratio,.20)
        rows[0]["global7_baseline_minus_strong_selfplay"]=-.5
        _,_,_,fail=x.primary_gates(records,rows)
        self.assertFalse(fail["stronger_attacker_heldout_transfer"])


if __name__=="__main__":
    unittest.main()
