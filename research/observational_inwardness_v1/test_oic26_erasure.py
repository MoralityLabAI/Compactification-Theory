"""Independent n6 one-erasure exact-policy and phase-symmetry tests."""
import base64,gzip,json,random,unittest
from oic23_exact import ExactFaultGame
from oic23_reference import replay_true_scenarios
from oic24_sym import SymFaultGame
from oic26_erasure import MANIFEST,ARCHIVE,check

class OIC26Tests(unittest.TestCase):
    def test_phase_translation_equivariant_at_six_phases(self):
        rng=random.Random(2606)
        g=SymFaultGame(6,11,0,1)
        for _ in range(240):
            state=(rng.getrandbits(36),rng.getrandbits(36))
            t=rng.randrange(11)
            m=rng.randrange(6)
            c=rng.randrange(6)
            rot=tuple(g.rotate_mask(v,c) for v in state)
            for y in (0,1,None):
                next_original=g.next_state(state,t,m,y)
                next_rotated=g.next_state(rot,t,(m+c)%6,y)
                expected=(tuple(g.rotate_mask(v,c) for v in next_original)
                          if next_original is not None else None)
                self.assertEqual(expected,next_rotated)

    def test_eleven_query_policy_against_independent_fault_paths(self):
        tree=json.loads(gzip.decompress(base64.b64decode(ARCHIVE.read_bytes().strip())))
        game=ExactFaultGame(6,11,0,1)
        self.assertEqual(game.verify_tree(tree),753)
        self.assertEqual(replay_true_scenarios(6,11,0,1,tree),331)

    def test_explicit_decision_tree_mutation_detected(self):
        tree=json.loads(gzip.decompress(base64.b64decode(ARCHIVE.read_bytes().strip())))
        def mutate(node):
            if 'shift' in node:
                node['shift']=(node['shift']+1)%6
            else:
                mutate(next(iter(node['children'].values())))
        mutate(tree)
        with self.assertRaises(AssertionError):ExactFaultGame(6,11,0,1).verify_tree(tree)

    def test_pinned_manifest_and_policy(self):
        m=check()
        self.assertEqual(m['minimum_adaptive_queries'],11)
        self.assertEqual(m['negative_unquotiented_states'],2223374)
        self.assertEqual(m['negative_translation_quotient_states'],370580)
        self.assertEqual(m['winning_translation_quotient_states'],27577)

    def test_constructive_witness_recomputed(self):
        g=SymFaultGame(6,11,0,1,node_cap=100000)
        result=g.solve()
        self.assertTrue(result)
        self.assertEqual(g.verify_tree(g.certificate()),753)

if __name__=='__main__':unittest.main()
