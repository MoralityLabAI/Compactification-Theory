import unittest
from gpu_dram_bench import parser, validate


class BenchGuardTests(unittest.TestCase):
    def test_default_is_cpu_resident(self):
        a = parser().parse_args([])
        validate(a)
        self.assertEqual((a.device, a.mode), ('cpu','resident'))

    def test_cuda_requires_deliberate_flag(self):
        with self.assertRaises(ValueError): validate(parser().parse_args(['--device','cuda']))

    def test_cpu_spill_is_not_gpu_evidence(self):
        with self.assertRaises(ValueError): validate(parser().parse_args(['--mode','spill']))

    def test_invalid_budgets_and_shapes(self):
        for argv in (['--loops','0'], ['--loops','5000'], ['--width','17'],
                     ['--max-seconds','nan'], ['--max-host-gib','-1'], ['--warmup','-1']):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                validate(parser().parse_args(argv))


if __name__ == '__main__': unittest.main()
