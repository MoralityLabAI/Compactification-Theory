import importlib.util
import unittest

from research.algernon.circuit_mirror.run import run_experiment


class TestSuite(unittest.TestCase):
    def test_reference_receipt(self):
        receipt, circuit = run_experiment(n=4, steps=5, cases_per_family=8, seed=10)
        self.assertEqual(receipt['exact_checks'], 32)
        self.assertEqual(receipt['exact_failures'], 0)
        self.assertGreater(receipt['negative_corrupt_trace_violations'],0)
        self.assertGreater(len(circuit.constraints),0)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'optional CPU torch')
    def test_learned_readout_pipeline_smoke(self):
        from research.algernon.circuit_mirror.pilot import train_and_evaluate
        report = train_and_evaluate(seed=7, n=4, updates=12, batch=16, cases_per_family=2)
        self.assertEqual(report['class'], 'toy_cpu_torch_supervised_proxy')
        self.assertEqual(report['parameter_count'], 4*4*4+4)
        self.assertEqual(len(report['results']),18)
        self.assertTrue(report['r1cs_oracle_example']['verification']['valid'])


if __name__ == '__main__': unittest.main()
