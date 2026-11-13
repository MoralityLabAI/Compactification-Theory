import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from knot import (t, burau, extract_closed_braid, normalized_alexander,
                  synthetic_braid, InvalidBraidTrace, braid_seifert_surface)


class ExactBraidTests(unittest.TestCase):
    def test_alexander_fixtures(self):
        cases = [
            (2, [1], 1),                  # unknot, Markov stabilization
            (2, [1,1,1], t**2 - t + 1),  # trefoil
            (3, [1,-2,1,-2], t**2 - 3*t + 1),  # figure eight
            (2, [1,1], t - 1),           # Hopf link, two components
            (2, [], 0),                  # two-component unlink
        ]
        for n, word, expected in cases:
            with self.subTest(n=n, word=word):
                self.assertEqual(normalized_alexander(n, word), expected)

    def test_braid_relation(self):
        self.assertEqual(burau(3, [1,2,1]), burau(3, [2,1,2]))
        self.assertEqual(burau(4, [1,3]), burau(4, [3,1]))
        self.assertEqual(burau(3, [1,-1]), burau(3, []))

    def test_markov_and_conjugation(self):
        trefoil = normalized_alexander(2, [1,1,1])
        self.assertEqual(trefoil, normalized_alexander(3, [1,1,1,2]))
        self.assertEqual(trefoil, normalized_alexander(3, [2,1,1,1,-2,2]))

    def test_canonical_seifert_surfaces(self):
        trefoil = braid_seifert_surface(2, [1, 1, 1])
        self.assertEqual((trefoil.boundary_components, trefoil.total_genus), (1, 1))
        self.assertEqual(braid_seifert_surface(3, [1, -2, 1, -2]).total_genus, 1)
        self.assertEqual(braid_seifert_surface(2, [1, 1]).total_genus, 0)
        unlink = braid_seifert_surface(2, [])
        self.assertEqual((unlink.surface_components, unlink.boundary_components,
                          unlink.total_genus), (2, 2, 0))
        # Same unknot boundary under braid cancellation, distinct surface genus.
        self.assertEqual(normalized_alexander(2, [1]),
                         normalized_alexander(2, [1, 1, -1]))
        self.assertEqual(braid_seifert_surface(2, [1]).total_genus, 0)
        self.assertEqual(braid_seifert_surface(2, [1, 1, -1]).total_genus, 1)

    def test_trace_matches_exact_words(self):
        for n, word in ((2,[1,1,1]), (3,[1,-2,1,-2]), (3,[1,2,-1,2]),
                        (2,[1,1]), (3, [])):
            with self.subTest(word=word):
                observed = extract_closed_braid(synthetic_braid(n,word))
                self.assertEqual(observed.word, tuple(word))
                self.assertEqual(normalized_alexander(n,observed.word),
                                 normalized_alexander(n,word))

    def test_open_endpoint_rejected(self):
        frames = synthetic_braid(2,[1,1])
        frames[-1][0] = (frames[-1][0][0]+.3, frames[-1][0][1])
        with self.assertRaisesRegex(InvalidBraidTrace, 'Open trace'):
            extract_closed_braid(frames)

    def test_unresolved_depth_rejected(self):
        frames = synthetic_braid(2,[1,1])
        frames = [[(x,0) for x,y in row] for row in frames]
        with self.assertRaisesRegex(InvalidBraidTrace, 'Unresolved'):
            extract_closed_braid(frames)

    def test_chart_affine_invariance(self):
        frames = synthetic_braid(3,[1,-2,1,-2])
        mapped = [[(3*x+10, 5*y-6) for x,y in row] for row in frames]
        self.assertEqual(extract_closed_braid(frames).word,
                         extract_closed_braid(mapped).word)


if __name__ == '__main__':
    unittest.main()

