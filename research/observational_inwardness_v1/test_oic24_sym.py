"""OIC-24 exact phase-translation quotient and robust n=5 witnesses."""
from __future__ import annotations
import base64,gzip,hashlib,json,random,unittest
from oic24_sym import SymFaultGame
from oic23_exact import ExactFaultGame
from oic23_reference import replay_true_scenarios
from oic24_run import MANIFEST,TREE

class OIC24Tests(unittest.TestCase):
    def test_equivariance_of_all_fault_successors(self):
        rng=random.Random(2311)
        for n in (2,3,4,5):
            for e,r in ((1,0),(0,1),(1,1)):
                g=SymFaultGame(n,12,e,r)
                for _ in range(100):
                    state=tuple(rng.getrandbits(n*n) for _ in range(g.width))
                    t=rng.randrange(12);m=rng.randrange(n);c=rng.randrange(n)
                    translated=tuple(g.rotate_mask(mask,c) for mask in state)
                    for y in ((0,1,None) if r else (0,1)):
                        left=g.next_state(state,t,m,y)
                        right=g.next_state(translated,t,(m+c)%n,y)
                        transformed=tuple(g.rotate_mask(mask,c) for mask in left) if left is not None else None
                        self.assertEqual(transformed,right)

    def test_exact_erasure_budget_at_five_phases(self):
        short=ExactFaultGame(5,8,0,1,node_cap=200000)
        self.assertFalse(short.solve())
        full=SymFaultGame(5,9,0,1,node_cap=200000)
        self.assertTrue(full.solve())
        tree=full.certificate()
        self.assertEqual(full.verify_tree(tree),341)
        self.assertEqual(replay_true_scenarios(5,9,0,1,tree),174)

    def test_one_flip_lower_bound_eleven_at_five_phases(self):
        game=SymFaultGame(5,11,1,0,node_cap=600000)
        self.assertFalse(game.solve())
        self.assertEqual(game.visited,567127)

    def test_pinned_certificate_and_manifest(self):
        m=json.loads(MANIFEST.read_text(encoding='utf8'))
        comp=base64.b64decode(TREE.read_bytes().strip(),validate=True)
        canonical=gzip.decompress(comp)
        self.assertEqual(hashlib.sha256(comp).hexdigest(),m['policy_gzip_sha256'])
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),m['policy_canonical_json_sha256'])
        tree=json.loads(canonical)
        self.assertEqual(replay_true_scenarios(5,9,0,1,tree),174)
        self.assertEqual(m['n'],5)
        self.assertEqual(m['erasure_only']['minimum_queries'],9)
        self.assertEqual(m['one_substitution']['known_impossible_through'],11)
        self.assertEqual(m['one_substitution']['Q12_status'],'inconclusive_700000_state_cap')

    def test_false_policy_mutation_is_rejected(self):
        cert=json.loads(gzip.decompress(base64.b64decode(TREE.read_bytes().strip())))
        def mutate(node):
            if 'shift' in node:
                node['shift']=(node['shift']+1)%5
                return True
            for child in node['children'].values():
                if mutate(child):return True
            return False
        self.assertTrue(mutate(cert))
        with self.assertRaises(AssertionError):
            replay_true_scenarios(5,9,0,1,cert)

if __name__=='__main__':unittest.main()
