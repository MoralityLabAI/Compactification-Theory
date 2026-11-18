"""Toy content hashing and sealing tests, independent of the NOR60 cohort."""
import copy
import unittest
from unittest.mock import patch
from research.nor60_authorized_interpreter import cohort


class PublicInputIntegrityTests(unittest.TestCase):
    def test_content_hash_is_independent_of_experiment_metadata(self):
        old = {"length": 5, "source_coefficients_decimal": ["1", "0", "-2", "0", "3"],
               "source_schema": "old", "source_sha256": "a" * 64, "profile": "alpha"}
        new = {"full_length": 5, "public_query_coefficients_decimal": ["1", "0", "-2", "0", "3"],
               "case_schema": "new", "source_sha256": "b" * 64, "program_profile": 7}
        self.assertEqual(cohort.query_content_sha256(old), cohort.query_content_sha256(new))
        edited = copy.deepcopy(new)
        edited["public_query_coefficients_decimal"][4] = "4"
        self.assertNotEqual(cohort.query_content_sha256(old), cohort.query_content_sha256(edited))
        for key, value in (("full_length", 6), ("public_query_coefficients_decimal", ["1", "00", "-2", "0", "3"]),
                           ("public_query_coefficients_decimal", ["1", 0, "-2", "0", "3"])):
            with self.assertRaises((ValueError, TypeError)):
                cohort.query_content_sha256(dict(new, **{key: value}))
        with self.assertRaises(ValueError):
            cohort.query_content_sha256({"length": 3, "unknown_vector": ["1", "0", "1"]})

    def test_all_historical_rows_are_accounted_for_without_new_source_generation(self):
        history = cohort.historical_evidence()
        self.assertEqual(history["source_row_count"], 588)
        self.assertEqual(len(history["source_ids"]), 588)
        self.assertEqual(len(history["query_content_ids"]), 588)
        self.assertEqual(len(history["program_input_ids"]), 60)
        self.assertEqual(len(history["archive_records"]), 10)

    def test_generation_is_sealed_without_registration_marker(self):
        with patch.object(cohort, "registration_sha", side_effect=ValueError("sealed")):
            with self.assertRaisesRegex(ValueError, "sealed"):
                cohort.make_fixture()
            with self.assertRaisesRegex(ValueError, "sealed"):
                cohort.source_series(512, 4, 0, 0)


if __name__ == "__main__":
    unittest.main()
