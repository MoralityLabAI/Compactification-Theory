"""NOR-08C: source-locked basin partition and intervention controls."""
import unittest
from dataclasses import replace

from research.nor01_numerical_orbit_sketch import nor08_basin_partition as probe
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as survey


class BasinPartitionTests(unittest.TestCase):
    def test_heldout_cohort_source_lock_and_no_duplicates(self):
        model,p1,p2=probe.locked_model()
        self.assertEqual(len(p1),12)
        self.assertEqual(len(p2),12)
        self.assertGreater(probe.distance(p1,p2),5.0)
        cases=probe.cohort(2)
        self.assertEqual(len(cases),12)
        self.assertEqual(len(set(c.visible for c in cases)),12)

    def test_future_truth_and_hidden_program_mutation(self):
        model,_,_=probe.locked_model()
        case=probe.cohort(1)[0]
        h,parity,heads=survey.prefix_hidden(case,model)
        changed=replace(case,program=None,future=(444444,)*len(case.future))
        h2,parity2,heads2=survey.prefix_hidden(changed,model)
        self.assertEqual(h,h2)
        self.assertEqual(parity,parity2)
        self.assertEqual(heads,heads2)

    def test_frozen_intervention_is_deterministic(self):
        a=probe.run(samples_per_family=1,steps=24)
        b=probe.run(samples_per_family=1,steps=24)
        self.assertEqual(a,b)
        self.assertEqual(a["cohort_size"],6)
        self.assertEqual(a["intervention"],"flip hidden coordinate 0 sign immediately after nine public input rows")
        self.assertFalse(a["source_future_labels_used"])
        self.assertFalse(a["new_training_performed"])
        self.assertFalse(a["exact_source_to_second_basin_capture_proved"])
        self.assertEqual(sum(a["coordinate_intervention_counts"].values()),6)
        self.assertLessEqual(a["unique_public_feature_tensors"],6)
        self.assertLessEqual(a["unique_native_initial_hidden_states"],6)
        self.assertEqual(len(a["per_family_input_diversity"]),6)
        for family,diversity in a["per_family_input_diversity"].items():
            self.assertEqual(diversity["distinct_visible"],1)
            self.assertGreaterEqual(diversity["distinct_features"],1)
            self.assertGreaterEqual(diversity["distinct_hidden"],1)

    def test_invalid_sampling_rejected(self):
        with self.assertRaises(ValueError):
            probe.cohort(0)
        with self.assertRaises(ValueError):
            probe.cohort(49)
        with self.assertRaises(ValueError):
            probe.run(samples_per_family=1,steps=10)
        with self.assertRaises(ValueError):
            probe.run(samples_per_family=1,steps=2001)


if __name__=="__main__":
    unittest.main()
