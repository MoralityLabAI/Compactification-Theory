import unittest
from research.shaft_compactification.trigger_identifiability import (
 V,FAMILIES,true_trigger,one,two,recover_row,pair_gauge_equal,reference,
)
class TriggerReferenceTests(unittest.TestCase):
 def test_matrix_structure(self):
  for family in FAMILIES:
   T=true_trigger(family)
   self.assertEqual(len(T),V)
   self.assertTrue(all(sum(row)==1 for row in T))
 def test_reconstruction(self):
  for family in FAMILIES:
   T=true_trigger(family)
   for p in range(V):
    for b in (0,15,33,63):self.assertEqual(recover_row(one(b),two(b,p,T)),T[p])
 def test_arbitrary_matrix(self):
  T=tuple(tuple((p*a+p+2*a)%2 for a in range(V)) for p in range(V))
  self.assertEqual(recover_row(one(29),two(29,4,T)),T[4])
 def test_unknown_previous_gauge(self):
  T=true_trigger('repeat_toggle');perm=(1,0,2,3,4,5)
  self.assertTrue(pair_gauge_equal(23,3,T,perm))
 def test_invalid(self):
  with self.assertRaises(ValueError):recover_row((0,),tuple(range(36)))
  with self.assertRaises(ValueError):true_trigger('unknown')
 def test_exhaustive_reference(self):
  r=reference()
  self.assertEqual(r['family_state_checks'],768)
  self.assertEqual(r['unlabelled_row_permutations_checked'],8640)
if __name__=='__main__':unittest.main()
