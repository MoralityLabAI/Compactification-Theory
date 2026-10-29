"""Finite provenance, coder-packet and transmission-graph gate checks."""
from __future__ import annotations

import copy
import csv
import tempfile
from pathlib import Path
import unittest

from phase2 import (AXES, load, graph_report, make_packets, score_code_files,
                    transmission_path, verify_graph, verify_passages)

HERE = Path(__file__).resolve().parent


class HistoricalGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = load(HERE / "influence_graph.json")

    def test_graph_validated(self):
        verify_graph(self.graph)
        self.assertEqual(len(self.graph["nodes"]), 16)
        self.assertEqual(len(self.graph["edges"]), 11)

    def test_supported_plotinus_to_falsafa_path(self):
        self.assertEqual(transmission_path(self.graph, "PLOTINUS", "FALSAFA"),
                         ["PLOTINUS", "ARABIC_PLOTINUS", "FALSAFA"])

    def test_supported_biblical_reception_to_witnesses(self):
        self.assertEqual(transmission_path(self.graph, "APOCALYPTIC_BIBLE", "JW_PRESENT"),
                         ["APOCALYPTIC_BIBLE", "BIBLE_STUDENTS", "JW_PRESENT"])

    def test_no_direct_or_mediated_known_jw_to_kurzweil(self):
        self.assertIsNone(transmission_path(self.graph, "BIBLE_STUDENTS", "KURZWEIL_2005"))
        self.assertIsNone(transmission_path(self.graph, "SHANKARA", "KURZWEIL_2005"))
        self.assertIsNone(transmission_path(self.graph, "DAO_DE_JING", "KURZWEIL_2005"))

    def test_analysis_edge_cannot_be_read_as_transmission(self):
        self.assertIsNone(transmission_path(self.graph, "KURZWEIL_2005", "GERACI_2010"))
        self.assertEqual(graph_report(self.graph)["edge_counts"]["scholarly_analysis:documented"], 3)

    def test_group_level_reception_has_explicit_scope(self):
        self.assertEqual(transmission_path(self.graph, "US_AI_FUTURISM", "INDIAN_AI_RECEPTION"),
                         ["US_AI_FUTURISM", "INDIAN_AI_RECEPTION"])
        e = next(e for e in self.graph["edges"] if e["to"] == "INDIAN_AI_RECEPTION")
        self.assertIn("not person-level", e["evidence_scope"])

    def test_dangling_edge_rejected(self):
        g = copy.deepcopy(self.graph)
        g["edges"][0]["to"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "dangling"):
            verify_graph(g)

    def test_unjustified_documented_edge_rejected(self):
        g = copy.deepcopy(self.graph)
        del g["edges"][0]["evidence_scope"]
        with self.assertRaisesRegex(ValueError, "source scope"):
            verify_graph(g)

    def test_chronological_impossibility_rejected(self):
        g = copy.deepcopy(self.graph)
        g["edges"][0]["from"] = "GERACI_2022"
        with self.assertRaisesRegex(ValueError, "chronologically impossible"):
            verify_graph(g)

    def test_documented_cycle_rejected(self):
        g = copy.deepcopy(self.graph)
        g["edges"].append({"from":"FALSAFA","to":"PLOTINUS", "edge_class":"transmission",
                           "status":"documented","mechanism":"fictional_backflow",
                           "evidence_url":"https://example.org/","evidence_scope":"fabricated"})
        with self.assertRaises(ValueError):
            verify_graph(g)


class CoderPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load(HERE / "passages.json")

    def test_passages_validated(self):
        verify_passages(self.data)
        self.assertEqual(len(self.data["passages"]), 11)

    def test_duplicate_passages_rejected(self):
        p = copy.deepcopy(self.data)
        p["passages"].append(copy.deepcopy(p["passages"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_passages(p)

    def test_synthetic_source_forgery_rejected(self):
        p = copy.deepcopy(self.data)
        p["passages"][-1]["source_url"] = "https://fake.example.org"
        with self.assertRaisesRegex(ValueError, "synthetic case"):
            verify_passages(p)

    def test_two_packets_exclude_hypothesis_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = make_packets(self.data, out)
            self.assertEqual(result["coded_rows"], 0)
            self.assertFalse(result["source_blinded"])
            self.assertTrue(result["pairing_blinded"])
            with (out / "coder_A.csv").open(encoding="utf-8", newline="") as f:
                rows_a = list(csv.DictReader(f))
            with (out / "coder_B.csv").open(encoding="utf-8", newline="") as f:
                rows_b = list(csv.DictReader(f))
            self.assertNotEqual([r["passage_key"] for r in rows_a], [r["passage_key"] for r in rows_b])
            self.assertEqual({r["passage_key"] for r in rows_a}, {r["passage_key"] for r in rows_b})
            self.assertTrue(all(not row["agency"] and not row["time"] for row in rows_a + rows_b))
            self.assertNotIn("researcher_note", rows_a[0])
            self.assertNotIn("source_id", rows_a[0])

    def test_uncoded_templates_have_no_fake_reliability(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            make_packets(self.data, out)
            result = score_code_files(out / "coder_A.csv", out / "coder_B.csv")
            self.assertEqual(result["coding_status"], "incomplete_annotations")
            self.assertEqual(result["axes"]["agency"]["eligible"], 0)
            self.assertIsNone(result["axes"]["agency"]["cohen_kappa"])
            self.assertFalse(result["human_independence_verified"])

    def test_perfect_agreement_with_two_categories(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            make_packets(self.data, out)
            for name in ("coder_A.csv", "coder_B.csv"):
                path = out / name
                with path.open(encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    label = "category_one" if int(row["passage_key"][1:]) % 2 else "category_two"
                    for axis in AXES:
                        row[axis] = label
                with path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            report = score_code_files(out / "coder_A.csv", out / "coder_B.csv")
            self.assertEqual(report["axes"]["agency"]["raw_agreement"], 1.0)
            self.assertEqual(report["axes"]["agency"]["cohen_kappa"], 1.0)
            self.assertFalse(report["human_independence_verified"])

    def test_same_file_cannot_be_two_coders(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            make_packets(self.data, out)
            with self.assertRaisesRegex(ValueError, "distinct"):
                score_code_files(out / "coder_A.csv", out / "coder_A.csv")


if __name__ == "__main__":
    unittest.main()
