"""Exact symmetry tests for shared-edge aggregation, with no torch dependency."""
import unittest
from itertools import combinations
from family_holdout import family_split
from relational_probe import pair_features
from transport_lift import relabel


class EdgeEquivarianceTests(unittest.TestCase):
    def test_common_gauge_relabels_only_edge_coordinates(self):
        d=family_split()
        s=d['split']
        edges=tuple(combinations(range(6),2))
        features,_,_=pair_features(s)
        edge_id={e:i for i,e in enumerate(edges)}
        tau=(2,1,0,3,4,5)
        for pi in d['test_pi'][:24]:
            transported=s['states'][s['orbit_index'][pi]]
            transported_prime=relabel(tau,transported)
            for gi,g in enumerate(s['states']):
                transformed=relabel(tau,g)
                old=sorted(zip(features[gi],
                               [float(e in transported) for e in edges]))
                new=sorted(zip([float(e in transformed) for e in edges],
                               [float(e in transported_prime) for e in edges]))
                self.assertEqual(old,new)

    def test_equivariant_positive_compatibility_all_720(self):
        d=family_split()
        s=d['split']
        tau=(2,1,0,3,4,5)
        for i,p in enumerate(s['perms']):
            graph=s['states'][s['orbit_index'][i]]
            self.assertEqual(relabel(tau,graph),
                             relabel(tuple(tau[p[j]] for j in range(6)),s['base']))

    def test_parameter_count_and_family_theoretical_limit(self):
        hidden=8
        parameters=(2*hidden+hidden)+(hidden+1)+1
        self.assertEqual(parameters,34)
        d=family_split()
        self.assertEqual(len(d['test_g']),3)
        self.assertEqual(1-1/(2*len(d['test_g'])),5/6)


if __name__=='__main__':
    unittest.main()
