"""Standard-library OIC-15 independent frozen receipt and tamper checks."""
import copy
import json
import unittest
from pathlib import Path

from oic15_verify import RESULTS, load_archive, validate_panels, verify


class OIC15EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((RESULTS/'oic15_manifest.json').read_text())
        cls.traces=load_archive(RESULTS/cls.manifest['files']['traces']['path'],cls.manifest['files']['traces'])
        cls.analysis=json.loads((RESULTS/'oic15_analysis.json').read_text())

    def test_stored_receipt_source_and_oracles(self):
        receipt=verify()
        self.assertEqual(receipt['fit_count'],30)
        self.assertEqual(receipt['verified_panels'],900)
        self.assertEqual(receipt['verified_aggregates'],120)

    def test_tampered_model_probability_rejected(self):
        sample=copy.deepcopy(self.traces)
        record=sample['runs'][0]['panels']['4']['latch']
        record['predicted_probabilities'][0]=1-record['predicted_probabilities'][0]
        with self.assertRaises(ValueError):
            validate_panels(sample,self.analysis)

    def test_tampered_initial_phase_interface_rejected(self):
        sample=copy.deepcopy(self.traces)
        sample['protocol']['observer_receives_phase']=True
        with self.assertRaises(ValueError):
            validate_panels(sample,self.analysis)

    def test_tampered_passive_signal_rejected(self):
        sample=copy.deepcopy(self.traces)
        sample['runs'][0]['panels']['4']['passive']['predicted_probabilities'][1]=0.13
        with self.assertRaises(ValueError):
            validate_panels(sample,self.analysis)


if __name__=='__main__':unittest.main()
