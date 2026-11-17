"""Pre-data NOR-38 tests use archived NOR-37 public sources, never the new cohort."""
from __future__ import annotations
import unittest
from collections import Counter
from fractions import Fraction
from math import comb
from research.nor38_global_uniform import experiment as x


class NOR38PreDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=x.history()
        cls.row=x.nor37.new_source(56,512,x.FAMILIES[0],set(x.nor37.history()))
        cls.public=x.nor37.public_proposal(cls.row)
        cls.src=cls.row["full_public_input_sha256"]
        cls.geometry=x.nor37.source_geometry(
            cls.public["pred"],cls.public["header"],cls.src)
        cls.risk,cls.cats,cls.union,cls.cu,cls.cv,cls.bu,cls.bv,cls.scores=cls.geometry

    def test_registration_and_original_archive_locks(self):
        self.assertEqual(len(self.old),180)
        self.assertEqual(len(set(self.old)),180)
        self.assertEqual(x.REPS,tuple(range(60,68)))
        self.assertEqual(x.OFFSET,130038000)
        self.assertEqual(x.REGISTERED,"37f9ed96b122ea16fb681dd5df638cbc4285d1e4")
        self.assertIn(self.src,self.old)
        self.assertEqual((x.K,x.R,x.CAL,x.EVAL,x.ATTACKS),(16,4,256,512,8))
        self.assertEqual(x.WEIGHTS,(2,0,1,1,4,8))
        self.assertEqual(len(x.POLICIES),7)
        self.assertEqual(len(x.CONTROLS),5)

    def test_first_public_source_has_no_private_leaks(self):
        r=self.row
        self.assertEqual(r["split"],"TEST")
        self.assertEqual(r["source_seed"],x.nor37.source_seed(56,512,x.FAMILIES[0]))
        self.assertTrue(all(k not in r for k in
                            ("program","future","target","attack","witness","audit")))
        self.assertEqual(r["original_native_public_mask"],self.public["mask"])
        self.assertEqual(len(self.public["pred"]),769)
        self.assertEqual(x.nor37.restore_native_target(r,self.public["pred"]),self.public["pred"])

    def test_mixture_exact_integer_weights(self):
        self.assertEqual(x.POLICIES,(
            "uniform16","original_mix","deterministic_mix","stochastic50_mix",
            "atom_random_h50","global7_mix","global8_mix"))
        for name in x.POLICIES:
            parts=x.components(name)
            self.assertEqual(sum(w for _,_,w in parts),256)
            self.assertTrue(all(w>0 and mode in x.MODES and arm in x.ARMS
                                for mode,arm,w in parts))
        self.assertEqual(x.components("uniform16"),(("original","uniform",256),))
        g7=x.components("global7_mix")
        g8=x.components("global8_mix")
        self.assertEqual(g7[0],("original","uniform",112))
        self.assertEqual(g8[0],("original","uniform",128))
        self.assertEqual([w for _,_,w in g7[1:]],[18,9,9,36,72])
        self.assertEqual([w for _,_,w in g8[1:]],[16,8,8,32,64])
        self.assertEqual(x.components("original_mix")[0],("original","uniform",32))

    def test_phase_domain_separation_and_quota_preservation(self):
        self.assertNotEqual(x.phase_tag(self.src,"cal"),x.phase_tag(self.src,"eval"))
        original=tuple(x.nor25.policy_schedules(
            self.risk,x.phase_tag(self.src,"cal"),3)[("hybrid_50",x.K)])
        self.assertEqual(len(original),x.K)
        quotas=tuple(sum(i in atom for i in original) for atom in self.cats)
        root=x.phase_tag(self.src,"cal")
        for mode in ("atom_random","atom_curvature50"):
            p=x.nor35.transformed_plan(original,self.cats,self.scores,
                                       "nor38-test", "hybrid_50",3,mode)
            self.assertEqual(len(set(p)),x.K)
            self.assertEqual(tuple(sum(i in atom for i in p) for atom in self.cats),
                             quotas)
        stoch,behavior=x.nor37.new_stochastic_plan(
            original,self.cats,self.scores,root,self.src,"hybrid_50",3,"stochastic50")
        self.assertEqual(len(set(stoch)),x.K)
        self.assertEqual(tuple(sum(i in atom for i in stoch) for atom in self.cats),
                         quotas)
        self.assertEqual(behavior["explore_atoms"]+behavior["targeted_atoms"],
                         sum(q>0 for q in quotas))

    def test_genuine_global_uniform_support_and_no_atom_quota_condition(self):
        m=len(self.public["pred"])
        self.assertGreater(m,len(self.union))
        # Entire output support, including positions outside original old-risk union.
        outside=set(range(m))-self.union
        self.assertGreater(len(outside),0)
        plan=x.nor25.policy_schedules(
            self.risk,x.phase_tag(self.src,"cal"),1)[("uniform",x.K)]
        self.assertEqual(len(plan),16)
        self.assertEqual(len(set(plan)),16)
        # New mixture must use an unconditioned existing full-uniform arm,
        # not a quota-preserving per-atom redraw.
        self.assertEqual(x.components("global7_mix")[0][:2],
                         ("original","uniform"))

    def test_full_output_exact_maximin_and_global_mixture_floor(self):
        for m in (769,1537):
            expected=Fraction(1)-Fraction(comb(m-16,4),comb(m,4))
            self.assertEqual(expected,x.nor35.ideal_uniform_attack_detect(m))
            self.assertGreater(expected,0)
            self.assertLess(expected,1)
            for num in (7,8):
                self.assertGreater(Fraction(num,16)*expected,0)
        # Exhaustive elementary finite counting validates the floor on a
        # smaller game, including attacks chosen with public policy knowledge.
        from itertools import combinations
        m,k,r=7,2,2
        draws=list(combinations(range(m),k))
        attacks=list(combinations(range(m),r))
        attack=attacks[0]
        rate=Fraction(sum(bool(set(q)&set(attack)) for q in draws),len(draws))
        self.assertEqual(rate,Fraction(1)-Fraction(comb(m-k,r),comb(m,r)))
        for attack in attacks:
            same=Fraction(sum(bool(set(q)&set(attack)) for q in draws),
                          len(draws))
            self.assertEqual(same,rate)
            for biased in (0,Fraction(1,3),1):
                combined=Fraction(7,16)*same+Fraction(9,16)*biased
                self.assertGreaterEqual(combined,Fraction(7,16)*rate)

    def test_separate_policy_aware_greedy_and_predeclared_blind_attacks(self):
        m=80
        fake={(mode,arm):[(1<<((j+len(mode)+len(arm))%29))
                          for j in range(m)] for mode in x.MODES for arm in x.ARMS}
        for name in x.POLICIES:
            attacks=[x.greedy_attack(fake,name,"a"*64,i) for i in range(8)]
            self.assertTrue(all(len(a)==len(set(a))==4 for a in attacks))
            self.assertTrue(all(all(0<=j<m for j in a) for a in attacks))
            self.assertEqual(attacks,[x.greedy_attack(fake,name,"a"*64,i)
                                      for i in range(8)])
        m=len(self.public["pred"])
        mock={(mode,arm):[1<<(j%31) for j in range(m)]
              for mode in x.MODES for arm in x.ARMS}
        attacks=x.blind_attacks(self.src,self.cats,self.union,self.cu,self.cv,
                                self.bu,self.bv,m,mock)
        self.assertEqual(set(attacks),set(x.CONTROLS))
        self.assertTrue(all(len(vals)==8 and
                            all(len(set(a))==4 for a in vals)
                            for vals in attacks.values()))
        for a,b in zip(attacks["top_curvature"],
                       attacks["eight_atom_matched"]):
            self.assertEqual([sum(i in cat for i in a) for cat in self.cats],
                             [sum(i in cat for i in b) for cat in self.cats])

    def test_independent_eval_diagnostic_branch(self):
        choices=[x.realized_component(self.src,t,"global7_mix")
                 for t in range(1000)]
        self.assertGreater(sum(m=="original" and a=="uniform" for m,a in choices),
                           250)
        self.assertGreater(sum(m=="atom_curvature50" for m,a in choices),250)
        self.assertEqual(choices,[x.realized_component(self.src,t,"global7_mix")
                                  for t in range(1000)])
        self.assertNotEqual(choices,
                            [x.realized_component(self.src,t,"global8_mix")
                             for t in range(1000)])


    def test_independent_replayer_on_archived_source_only(self):
        from research.nor38_global_uniform import receipt_audit as audit
        old,atoms,u,cu,cv,bu,bv,scores=audit.geometry(
            self.public["pred"],self.public["header"])
        self.assertEqual((list(map(len,atoms)),u,cu,cv,bu,bv),
                         (list(map(len,self.cats)),self.union,
                          self.cu,self.cv,self.bu,self.bv))
        self.assertEqual(scores,list(self.scores))
        for policy in x.POLICIES:
            self.assertEqual(x.components(policy),audit.component_parts(policy))
        cal,diag=x.schedules(self.risk,self.cats,self.scores,self.src,"cal")
        separate,separate_diag=audit.policy_schedules(
            old,atoms,scores,self.src,"cal")
        self.assertEqual(x.canonical(cal),audit.canonical(separate))
        self.assertEqual(diag,separate_diag)
        cov=x.trial_exposure(cal,len(self.public["pred"]))
        greedy,visits=audit.independent_greedy_attacks(
            separate,self.src,len(self.public["pred"]))
        for policy in x.POLICIES:
            self.assertEqual(
                [x.greedy_attack(cov,policy,self.src,t) for t in range(x.ATTACKS)],
                greedy[policy])
        blind=x.blind_attacks(self.src,self.cats,self.union,self.cu,self.cv,
                              self.bu,self.bv,len(self.public["pred"]),cov)
        independent_blind=audit.independent_blind(
            self.src,atoms,u,cu,cv,bu,bv,len(self.public["pred"]),visits)
        self.assertEqual({k:[list(a) for a in v] for k,v in blind.items()},
                         {k:[list(a) for a in v] for k,v in independent_blind.items()})
        ev,_=x.schedules(self.risk,self.cats,self.scores,self.src,"eval")
        independent_eval,_=audit.policy_schedules(
            old,atoms,scores,self.src,"eval")
        self.assertEqual(x.canonical(ev),audit.canonical(independent_eval))
        self.assertEqual(x.exact_top(ev,self.cu,self.cv),
                         audit.exact_top(independent_eval,cu,cv))
        for policy in ("global7_mix","global8_mix"):
            self.assertEqual(
                [x.realized_component(self.src,t,policy) for t in range(x.EVAL)],
                [audit.realized_choice(self.src,t,policy) for t in range(x.EVAL)])

    def test_gates_built_at_48_source_level(self):
        rows=[{
            "global7_self_minus_deterministic_self":.015,
            "global7_worst_minus_original_worst":-.004,
            "global7_top_minus_atom_random":.035,
            "deterministic_top_minus_atom_random":.10,
            "global7_worst":.031,"replicate":60+i//6}
            for i in range(48)]
        good=[{
            "native_mask":1,"public_prediction_sha256":"a",
            "private_native_truth_sha256_after_freeze":"a",
            "honest_target_false_rejections":0,
            "fully_schedule_informed_attacker_detections":0,
            "delta_decision_disagreements":0,"direct_authentic_witness_checks":1,
            "exact_quota_and_control_checks":{"x":0},
            "exact_full_uniform_attack_detection_fraction":[1,10],
            "global7_exact_theoretical_floor":[1,20],
            "public_sha256":str(i)} for i in range(48)]
        means,wins,ratio,gates=x.primary_gates(good,rows)
        self.assertTrue(all(gates.values()))
        self.assertEqual(wins["global7_self_minus_deterministic_self"],48)
        self.assertGreater(ratio,.20)
        rows[0]["global7_worst_minus_original_worst"]=-1
        _,_,_,new=x.primary_gates(good,rows)
        self.assertFalse(new["global7_competitive_vs_original_mix_worst"])


if __name__=="__main__":
    unittest.main()
