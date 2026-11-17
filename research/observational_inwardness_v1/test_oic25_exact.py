"""Independent finite OIC-25 sensor-query and certificate integrity tests."""
from __future__ import annotations
import base64,gzip,json,unittest
from oic23_exact import ExactFaultGame
from oic23_reference import replay_true_scenarios
from oic25_independent import PythonPrunedGame
from oic25_reference import ARCHIVE,MANIFEST,canonical_bytes,sha_bytes

class OIC25Tests(unittest.TestCase):
    def test_pairwise_future_disagreement_formula(self):
        for Q in (5,7,13):
            game=PythonPrunedGame(Q)
            for a in range(25):
                for b in range(25):
                    for t in range(Q+1):
                        expected=sum((a%5+s*(a//5))%5!=(b%5+s*(b//5))%5 for s in range(t,Q))
                        self.assertEqual(game._future_diff[t][a][b],expected)

    def test_canonical_phase_translation_preserves_shift_labels_and_successors(self):
        from random import Random
        random=Random(2525)
        game=PythonPrunedGame(8)
        for _ in range(400):
            state=(random.getrandbits(25),random.getrandbits(25))
            c=random.randrange(5)
            translated=(game.rotated(state[0],c),game.rotated(state[1],c))
            self.assertEqual(game.labels(state),game.labels(translated))
            t=random.randrange(8); m=random.randrange(5)
            for y in (0,1):
                a=game.next_state(state,t,m,y)
                b=game.next_state(translated,t,(m+c)%5,y)
                if a is None:self.assertIsNone(b)
                else:self.assertEqual((game.rotated(a[0],c),game.rotated(a[1],c)),b)

    def test_no_winning_policy_with_twelve_queries(self):
        game=PythonPrunedGame(12,node_cap=500000)
        self.assertFalse(game.solve())
        self.assertEqual(game.visited,125216)

    def test_winning_policy_with_fourteen_queries(self):
        game=PythonPrunedGame(14,node_cap=500000)
        self.assertTrue(game.solve())
        tree=game.certificate()
        independent=ExactFaultGame(5,14,1,0)
        self.assertEqual(independent.verify_tree(tree),662)
        self.assertEqual(replay_true_scenarios(5,14,1,0,tree),327)

    def test_pinned_policy_integrity_and_false_leaf(self):
        summary=json.loads(MANIFEST.read_text())
        comp=base64.b64decode(ARCHIVE.read_bytes().strip(),validate=True)
        raw=gzip.decompress(comp)
        self.assertEqual(sha_bytes(raw),summary['policy_canonical_json_sha256'])
        self.assertEqual(sha_bytes(comp),summary['policy_gzip_sha256'])
        tree=json.loads(raw)
        game=ExactFaultGame(5,14,1,0)
        self.assertEqual(game.verify_tree(tree),662)
        def mutate(node):
            if 'shift' in node:
                node['shift']=(node['shift']+1)%5
                return
            mutate(next(iter(node['children'].values())))
        mutate(tree)
        with self.assertRaises(AssertionError):game.verify_tree(tree)

if __name__=='__main__':unittest.main()
