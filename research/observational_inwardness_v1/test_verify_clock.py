"""Standard-library tests for the OIC-13 archived CPU structural-clock pilot."""
import copy
import unittest
from pathlib import Path

from verify_clock import ARCHIVE, decode_archive, validate_payload, verify


class OIC13ReceiptTests(unittest.TestCase):
    def test_receipt_and_source_digests(self):
        manifest=verify()
        self.assertEqual(manifest['model_fits'],120)
        self.assertEqual(manifest['pair_runs'],60)
        self.assertEqual(len(manifest['summary']),12)

    def test_oracle_tamper_detection(self):
        archived,_,_=decode_archive(ARCHIVE.read_bytes())
        bad=copy.deepcopy(archived)
        bad['runs'][0]['evaluations']['latch']['families']['fixed']['exact_state_traces'][0][1]^=1
        with self.assertRaises(ValueError):
            validate_payload(bad)

    def test_learned_trace_tamper_detection(self):
        archived,_,_=decode_archive(ARCHIVE.read_bytes())
        bad=copy.deepcopy(archived)
        bad['runs'][0]['evaluations']['latch']['families']['fixed']['predicted_state_traces'][0][1]^=1
        with self.assertRaises(ValueError):
            validate_payload(bad)

    def test_phase_split_tamper_detection(self):
        archived,_,_=decode_archive(ARCHIVE.read_bytes())
        bad=copy.deepcopy(archived)
        bad['runs'][0]['training']['fixed']['train_phase_indices']=[0]
        with self.assertRaises(ValueError):
            validate_payload(bad)

    def test_architecture_and_heldout_reconstruction(self):
        archived,_,_=decode_archive(ARCHIVE.read_bytes())
        summaries=validate_payload(archived)
        for row in summaries:
            self.assertEqual(row['models'],10)
            if row['architecture']=='reflection_tied':
                self.assertEqual(row['trainable_parameters_per_model'],2)
                self.assertEqual(row['all_phase_faithful_runs'],5)
            else:
                self.assertEqual(row['trainable_parameters_per_model'],3)
            if row['split']=='heldout' and row['architecture']=='oriented':
                self.assertEqual(row['latch_unseen_fixed'],0.5)
                self.assertEqual(row['latch_unseen_rotating'],1)


if __name__=='__main__':unittest.main()
