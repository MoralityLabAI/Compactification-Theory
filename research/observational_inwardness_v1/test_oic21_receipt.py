"""OIC-21 archive and source-integrity smoke tests, standard library only."""
import unittest
from pathlib import Path
from oic21_verify import report,read_json,read_compressed,HERE,RES,check

class OIC21Receipts(unittest.TestCase):
    def test_full_receipt_replay(self):
        r=report(silent=True)
        self.assertEqual(r['trained_fits'],10)
        self.assertEqual(r['fault_panels'],30)
    def test_file_hash_failure_is_fatal(self):
        with self.assertRaises(ValueError):check(False,'simulated hash mismatch')
    def test_phase_population_sizes(self):
        j=read_json(RES/'oic21_bench.json')
        self.assertEqual([x['n'] for x in j['results']],[4,5,6,7,8,12,16,24,32])
        for row in j['results']:
            self.assertEqual(row['query_budget'],row['n']+2)
    def test_model_weight_coverage(self):
        j=read_compressed(RES/'oic21_fits.json.gz.b64')
        self.assertEqual(len(j['fits']),10)
        for row in j['fits']:
            self.assertEqual(set(row['weights']),{'network.0.weight','network.0.bias','network.2.weight','network.2.bias'})

if __name__=='__main__':unittest.main()
