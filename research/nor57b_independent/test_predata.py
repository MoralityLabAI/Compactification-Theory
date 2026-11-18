"""NOR57B predata controls: no new NOR57B source generation or timing.

Check 408 prior original source SHA exclusions and repaired NOR57 reporter,
and reuse the existing unchanged signed-source exact attack tests.
"""
from __future__ import annotations
import unittest
from research.nor57_authenticated_router import accept
from research.nor57b_independent import benchmark_cpu,cohort

class IndependentProspectiveNOR57BPredata(unittest.TestCase):
    def test_frozen_source_and_selector_provenance(self):
        self.assertEqual(cohort.REGISTERED,
                         "0a9467ae18df3296d8fdf04971f0ffbbb4c6e3d7")
        self.assertEqual(cohort.SOURCE_DOMAIN,"NOR57B/SOURCE/v1")
        self.assertEqual(len(cohort.historical_ids()),408)
        self.assertEqual(len(cohort.REPS)*len(cohort.LENGTHS)*
                         len(cohort.PROFILES)*len(cohort.FAMILIES),60)
        self.assertEqual(accept.RULE_ID,"NOR57-FROZEN-PUBLIC-SELECTOR-v1")
        self.assertEqual(benchmark_cpu.REPEATS,5)
        self.assertEqual(benchmark_cpu.SCHEMA,
                         "nor57b.first_signed_v2_public_source_selector_cpu.v1")
        self.assertEqual(
            benchmark_cpu.MODE_BY_ARM[benchmark_cpu.ARMS[2]],
            "frozen_selector")

    def test_repaired_source_reporter_from_original_negative_result(self):
        fake=[
            {"frozen_public_selector_decision":route,
             "always_native_over_always_GMP":.70,
             "frozen_selector_over_always_GMP":.75 if route==accept.NATIVE else 1.02,
             "frozen_selector_over_always_native":1.1,
             "arm_medians_ms":{
                 benchmark_cpu.ARMS[0]:2.0,
                 benchmark_cpu.ARMS[1]:1.4,
                 benchmark_cpu.ARMS[2]:1.5 if route==accept.NATIVE else 2.04}
            } for route in (accept.NATIVE,accept.GMP)]
        summary=benchmark_cpu.summarize(fake)
        self.assertEqual(summary["count"],2)
        self.assertEqual(summary["selector_native_case_count"],1)
        self.assertEqual(summary["selector_gmp_case_count"],1)
        self.assertEqual(summary["sum_source_median_ms_by_arm"][benchmark_cpu.ARMS[0]],4.0)
        self.assertAlmostEqual(summary["sum_source_median_ms_by_arm"][benchmark_cpu.ARMS[2]],
                               3.54)

    def test_original_first_failure_is_still_unscored(self):
        from research.nor57_authenticated_router import original_failure_audit
        first=original_failure_audit.read_original()
        self.assertEqual(first["original_outcome"],
                         "ENGINEERING_FAIL_REPORTING_EXCEPTION")
        self.assertEqual(first["first_executed_source_log_count"],60)
        self.assertFalse(first["first_CPU_json_written"])
        self.assertFalse(first["registered_five_gates_scored"])

if __name__=="__main__":
    unittest.main()
