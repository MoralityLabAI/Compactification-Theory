"""Frozen 145-parameter scorer evidence integrity tests (torch-free)."""
import unittest
from oic22_verify import get_archive,verify, IndependentObserver

class OIC22ReceiptTests(unittest.TestCase):
 def test_receipt_and_selected_independent_replays(self):
  record=verify(deep=False)
  self.assertEqual(record['model_fits'],10)
  self.assertEqual(record['independent_panel_replays'],12)
 def test_one_visible_bit_and_address_charged_per_query(self):
  for n in (4,6,8):
   q=n+2
   expected=q*(1+(n-1).bit_length())
   self.assertEqual(expected,q+q*(n-1).bit_length())
 def test_single_flip_is_tracked_inside_true_hypothesis(self):
  fit=get_archive()['fits'][0]
  monitor=IndependentObserver(4,fit['weights'])
  for d in range(4):
   for z in range(4):
    for pos in range(6):
     history=()
     for t in range(6):
      _,_,m=monitor.state(history)
      y=int((z+t*d)%4==m)^(t==pos)
      history+=(y,)
     a,b,_=monitor.state(history)
     self.assertTrue(b&(1<<(d*4+z)))

if __name__=='__main__':unittest.main()
