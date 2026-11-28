import unittest
from research.shaft_compactification.finite_observability import (
    DIM, STATES, HORIZONS, shift, trace, observability_matrix, gramian, reference,
)
class ShiftObservabilityTests(unittest.TestCase):
    def test_exhaustive_horizon_classes(self):
        for horizon in HORIZONS:
            groups={trace(s,horizon) for s in STATES}
            self.assertEqual(len(groups),2**min(horizon+1,DIM))
    def test_integer_gramian_identity_all_states(self):
        for horizon in HORIZONS:
            matrix=observability_matrix(horizon)
            G=gramian(horizon)
            for s in STATES:
                self.assertEqual(sum(s[i]*G[i][j]*s[j]
                                     for i in range(DIM) for j in range(DIM)),
                                 sum(sum(row[j]*s[j] for j in range(DIM))**2
                                     for row in matrix))
    def test_kernel_dimension_drops_one_per_step(self):
        self.assertEqual([x['kernel_dimension'] for x in reference()['results']],
                         [5,4,3,2,1,0,0,0,0])
    def test_deferred_coordinate_visible_only_after_five(self):
        zero=(0,)*DIM
        hidden=(0,0,0,0,0,1)
        for horizon in range(5):
            self.assertEqual(trace(zero,horizon),trace(hidden,horizon))
        self.assertNotEqual(trace(zero,5),trace(hidden,5))
    def test_full_horizon_distinguishes_all_states(self):
        self.assertEqual(len({trace(s,5) for s in STATES}),len(STATES))
    def test_rank_bound_on_rows(self):
        for horizon in HORIZONS:
            self.assertLessEqual(reference()['results'][horizon]['exact_observability_rank'],
                                 min(DIM,horizon+1))
if __name__=='__main__':unittest.main()
