"""NOR-30 registration, optimizer, public-only source and witness controls."""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations
from math import comb
import unittest

from research.nor30_minimax_audits import train as t
from research.nor30_minimax_audits import experiment as n
from research.nor21_degree_scaling import experiment as old
from research.nor24_adaptive_audits import experiment as prior24


class NOR30ProspectiveTests(unittest.TestCase):
    def test_training_uses_only_original_sha_locked_historical_game(self):
        receipt=t.freeze_training()
        self.assertEqual(receipt["schema"],t.SCHEMA)
        self.assertEqual(receipt["historical_source_count"],12)
        self.assertEqual(receipt["historical_denominator_per_cell"],49152)
        self.assertEqual(len(receipt["historical_integer_hit_matrix"]),7)
        self.assertEqual(len(receipt["historical_public_hashes"]),12)
        self.assertFalse(receipt["test_information_used_for_training"])
        w=receipt["constrained_solution"]["weights"]
        self.assertEqual(sum(w),16)
        self.assertGreaterEqual(w[0],2)
        self.assertLessEqual(w[0],12)
        self.assertEqual(t.optimize(receipt["historical_integer_hit_matrix"]),
                         receipt["constrained_solution"])

    def test_exact_grid_minimax_and_tie_break(self):
        # Every attacker sees the same payoff. The requested uniform
        # restriction is the only reason this optimum is non-pure.
        matrix=[[10,0,0,0,0,0,0] for _ in range(7)]
        constrained=t.optimize(matrix,True)
        pure=t.optimize(matrix,False)
        self.assertEqual(constrained["weights"],[12,0,0,0,0,0,4])
        self.assertEqual(pure["weights"],[16,0,0,0,0,0,0])
        # Two crossed strategies force the maximin to mix when allowed.
        crossover=[[0,16]+[0]*5,[16,0]+[0]*5]+[[16,0]+[0]*5 for _ in range(5)]
        fitted=t.optimize(crossover,False)
        self.assertEqual(fitted["weights"][0],8)
        self.assertEqual(fitted["weights"][1],8)

    def test_public_only_new_source_is_disjoint_and_native_exact(self):
        h=t.freeze_training()
        row=n.draw_source(20,512,n.FAMILIES[0],set())
        digest=row["full_public_input_sha256"]
        self.assertNotIn(digest,h["historical_public_hashes"])
        self.assertNotIn("program",row)
        self.assertNotIn("future",row)
        self.assertNotIn("witness",row)
        self.assertNotIn("target",row)
        public,header=prior24.public_only_case(row)
        self.assertIsNone(public.program)
        proposal=prior24.public_proposal(row)
        self.assertEqual(old.native_mask(public,header,512),
                         row["original_public_accept_mask"])
        self.assertEqual(proposal["mask"],row["original_public_accept_mask"])
        true=n.freeze_private_program(row,proposal["predicted"])
        self.assertEqual(true,proposal["predicted"])

    def test_exact_witness_compares_both_nonzero_shifts(self):
        future=(1,2,3,5,8,13,21,34)
        self.assertFalse(n.verify_hit(future,future,(1,2,3),(6,7)))
        self.assertTrue(n.verify_hit(future,future,(1,7),(6,7)))
        self.assertFalse(n.verify_hit(future,future,(),(6,7)))
        self.assertEqual(n.R,4)
        self.assertEqual(n.K,16)
        self.assertEqual(n.DELTAS,(1,4294049777))

    def test_ideal_uniform_and_hybrid_minimax_floors(self):
        for m in (769,1537):
            pure=n.ideal_floor(m,"uniform")
            self.assertEqual(pure,Fraction(1)-Fraction(comb(m-4,16),comb(m,16)))
            for name,u in (("hybrid_25",4),("hybrid_50",8),("hybrid_75",12)):
                self.assertEqual(n.ideal_floor(m,name),
                    Fraction(1)-Fraction(comb(m-(16-u)-4,u),comb(m-(16-u),u)))
            self.assertGreater(n.ideal_floor(m,"uniform"),n.ideal_floor(m,"hybrid_50"))
            self.assertEqual(n.ideal_floor(m,"static_prior"),0)
        # Exhaustively count a finite 8-position attack floor.
        m=8;k=3;r=2
        all_subsets=list(combinations(range(m),k))
        for attacked in combinations(range(m),r):
            hit=sum(bool(set(aud)&set(attacked)) for aud in all_subsets)
            self.assertEqual(Fraction(hit,len(all_subsets)),
                Fraction(1)-Fraction(comb(m-r,k),comb(m,k)))

    def test_greedy_mixture_uses_all_arms_and_unique_coordinates(self):
        m=20
        plans={a:[(j,(j+2)%m,(j+7)%m) for j in range(8)] for a in n.ARMS}
        cov={a:n.exposure(plans[a],m) for a in n.ARMS}
        w=[8,0,0,0,0,8,0]
        for name in n.ATTACKERS:
            for rep in range(2):
                att=n.best_response(cov,w,"a"*64,name,rep)
                self.assertEqual(len(att),n.R)
                self.assertEqual(len(set(att)),n.R)
                self.assertEqual(att,n.best_response(cov,w,"a"*64,name,rep))

    def test_training_and_test_phase_separation(self):
        self.assertNotEqual(n.phase_digest("a"*64,"calibration"),
                            n.phase_digest("a"*64,"evaluation"))
        self.assertNotEqual(n.seed(n.STREAM_ATTACK,"a"*64,0),
                            n.seed(n.STREAM_EVAL,"a"*64,0))
        weights=[4,0,0,0,0,12,0]
        picks=[n.mix_choose(weights,"a"*64,i) for i in range(256)]
        self.assertTrue(set(picks)<={"uniform","hybrid_50"})
        self.assertIn("uniform",picks)
        self.assertIn("hybrid_50",picks)


if __name__=="__main__":
    unittest.main()
