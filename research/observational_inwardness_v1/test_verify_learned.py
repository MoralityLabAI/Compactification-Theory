"""Standard-library tests for OIC-12 frozen supervised GRU receipt integrity."""
import copy
import unittest
from pathlib import Path

from verify_learned import DEFAULT_ARCHIVE, decode_archive, validate_payload, verify


class OIC12ReceiptTests(unittest.TestCase):
    def test_complete_receipt_and_source_digest(self):
        manifest = verify()
        self.assertEqual(manifest['model_fits'], 60)
        self.assertEqual(manifest['run_count'], 30)
        self.assertEqual(len(manifest['summary']), 6)

    def test_raw_trace_tampering_is_detected(self):
        archive, _, _ = decode_archive(DEFAULT_ARCHIVE.read_bytes())
        trial = copy.deepcopy(archive)
        words = trial['runs'][0]['evaluations']['latch']['families']['fixed']['predicted_state_traces']
        words[0][1] ^= 1
        with self.assertRaises(ValueError):
            validate_payload(trial)

    def test_exact_even_and_odd_reference_in_archive(self):
        archive, _, _ = decode_archive(DEFAULT_ARCHIVE.read_bytes())
        for record in archive['runs']:
            n = record['n']
            latch = record['evaluations']['latch']
            if n % 2 == 0:
                self.assertEqual(latch['oracle_code_distance'], n + 1)
                self.assertFalse(latch['oracle_terminal_overlap'])
            else:
                self.assertTrue(latch['oracle_terminal_overlap'])

    def test_phase_split_is_enforced(self):
        archive, _, _ = decode_archive(DEFAULT_ARCHIVE.read_bytes())
        trial = copy.deepcopy(archive)
        trial['runs'][0]['training']['fixed']['train_phase_indices'] = [0]
        with self.assertRaises(ValueError):
            validate_payload(trial)


if __name__ == '__main__':
    unittest.main()
