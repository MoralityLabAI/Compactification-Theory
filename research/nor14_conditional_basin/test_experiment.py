"""NOR-14: finite-design projection, causal readout and public-only trace tests."""
from __future__ import annotations

from dataclasses import replace
import math
import random
import unittest

from research.nor14_conditional_basin import experiment as n
from research.rrc_compactification import bench25, compound25 as corpus, model25


def make_toy_train():
    rng = random.Random(2026111514)
    rows = []
    for i in range(n.TRAIN_ROWS):
        z = [rng.uniform(-1, 1) for _ in range(12)]
        g = math.tanh(.7*z[0] + .3*z[1] - .5*z[2] + .4)
        rows.append({
            "x": {"hidden_only": z + [0., 1.]},
            "base": [0.02*(j % 4) for j in range(16)],
            "visible_mask": tuple(j in (i % 16, (i+3) % 16)
                                  for j in range(16)),
            "nor14_gate": {
                "endpoint_basin": g,
                "endpoint_shuffled": math.tanh(.6*z[3] - .2*z[5]),
                "endpoint_random": math.tanh(.3*z[4] + .4*z[6]),
                "trajectory_basin": math.tanh(.5*z[7] + .2*z[8]),
                "trajectory_random": math.tanh(.3*z[9] - .3*z[10]),
                "hidden_only": 0.,
            }
        })
    return rows


class NOR14PublicProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = make_toy_train()

    def test_orthogonalised_nonlinear_columns_have_zero_train_linear_moment(self):
        for kind in n.KINDS:
            p = n.fit_projection(self.rows, kind)
            self.assertLessEqual(p["linear_rank"], 13)
            self.assertLess(p["max_train_linear_moment"], 1e-7)
            self.assertEqual(len(p["coefficients"]), 12)
            self.assertEqual(len(p["normalizers"]), 12)
            self.assertTrue(all(v >= .05 for v in p["normalizers"]))
            for k in range(12):
                feat = [n.feature_vector(row, p)[12+k] for row in self.rows]
                for j in range(13):
                    moment = sum(a*n.linear_features(r)[j]
                                 for a, r in zip(feat, self.rows, strict=True)
                                 ) / len(feat)
                    self.assertLess(abs(moment), 1e-5)

    def test_nonzero_basin_gate_does_not_create_new_raw_information(self):
        row = self.rows[0]
        p = n.fit_projection(self.rows, "endpoint_basin")
        x = n.feature_vector(row, p)
        self.assertEqual(len(x), 25)
        self.assertEqual(x[:12], n.linear_features(row)[:12])
        self.assertEqual(x[24], 1.)
        self.assertTrue(all(math.isfinite(v) for v in x))
        stripped = n.feature_vector(row, p, remove_residual=True)
        self.assertEqual(stripped[12:24], [0.]*12)
        self.assertNotEqual(x[12:24], stripped[12:24])

    def test_fixed_matrix_can_react_to_only_one_gate_replacement(self):
        model = {
            "projection": n.fit_projection(self.rows, "endpoint_basin"),
            "weights": [[0.]*25 for _ in range(16)]
        }
        model["weights"][0][12] = 1.
        row = self.rows[0]
        first = n.raw_scores(row, model, .5)
        changed = n.raw_scores(row, model, .5, substitute=-1.)
        self.assertNotEqual(first[0], changed[0])
        self.assertEqual(first[1:], changed[1:])
        with self.assertRaises(ValueError):
            n.raw_scores(row, model, .5, substitute=2.)

    def test_public_softmax_gradient_and_short_fit(self):
        fitted = n.fit(self.rows, "endpoint_basin", epochs=2)
        self.assertEqual(fitted["slots"], 400)
        self.assertEqual(fitted["training_full_public_template_verifications_per_head"],1536)
        self.assertFalse(fitted["private_future_truth_used_in_fit"])
        self.assertTrue(all(math.isfinite(x) for w in fitted["weights"] for x in w))
        loss, grad = n.training_gradient(self.rows[0], fitted)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(len(grad), 16)
        self.assertEqual(len(grad[0]), 25)
        self.assertAlmostEqual(sum(x[24] for x in grad), 0., places=9)

    def test_group_sign_uses_distinct_nonlate_inputs(self):
        a = [
            {"public_features_sha256":"x","family":"signed","future_correct":0},
            {"public_features_sha256":"y","family":"rational","future_correct":1},
            {"public_features_sha256":"late","family":"late_source","future_correct":0},
            {"public_features_sha256":"late","family":"late_source","future_correct":0},
        ]
        b = [
            {**a[0],"future_correct":1},
            {**a[1],"future_correct":0},
            {**a[2],"future_correct":1},
            {**a[3],"future_correct":1},
        ]
        primary = n.paired(a, b)
        self.assertEqual(primary["cases"], 2)
        self.assertEqual(primary["unique_public_inputs"], 2)
        self.assertEqual(primary["paired_gains"], 1)
        self.assertEqual(primary["paired_losses"], 1)
        all_inputs = n.paired(a, b, nonlate_only=False)
        self.assertEqual(all_inputs["cases"], 4)
        self.assertEqual(all_inputs["unique_public_inputs"], 3)
        self.assertEqual(all_inputs["positive_unique_groups"], 2)

    def test_finite_trace_matches_original_native_logits_and_is_public_only(self):
        checkpoint, digests = bench25.checkpoints()
        ck = checkpoint["recurrent"]
        self.assertEqual(digests["recurrent"], n.old.BASIN_SHA)
        rng = random.Random(2026111514)
        for trial in range(30):
            case = corpus.make_case(rng,"signed",f"nor14-public-{trial}")
            headers = corpus.possible_headers(case)
            if len(headers) == 1:
                break
        else:
            self.fail("could not generate a unique public header")
        early, final = n.trace_at_four(case,headers[0],ck)
        self.assertEqual(len(early),12)
        self.assertEqual(len(final),12)
        native = model25.logits(case,headers[0],ck)
        from_trace = model25.affine(ck["state_dict"]["head.weight"],
                                   ck["state_dict"]["head.bias"],final)
        self.assertLess(max(abs(a-b) for a,b in zip(native,from_trace,strict=True)),1e-11)
        modified = replace(case,program=None,future=tuple(34567 for _ in case.future))
        early2, final2 = n.trace_at_four(modified,headers[0],ck)
        self.assertEqual(early,early2)
        self.assertEqual(final,final2)
        self.assertEqual(n.prior_probe.public_digest(case,headers[0]),
                         n.prior_probe.public_digest(modified,headers[0]))

    def test_fixed_sample_budget_rejects_excess_and_fit_rejects_changed_size(self):
        with self.assertRaises(ValueError):
            n.fit_projection(self.rows[:10],"endpoint_basin")
        with self.assertRaises(ValueError):
            n.fit(self.rows[:95],"endpoint_basin")
        with self.assertRaises(ValueError):
            n.fit(self.rows,"endpoint_basin",epochs=n.EPOCHS+1)
        with self.assertRaises(ValueError):
            n.fresh_cohorts({},repetitions=5)


if __name__ == "__main__":
    unittest.main()
