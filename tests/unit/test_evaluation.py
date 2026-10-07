#!/usr/bin/env python3
"""Check dataset integrity and metric semantics without native dependencies."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from evaluation import SAMPLE_FIELDS, load_dataset, load_legacy_features, sha256, summarize


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="evaluation-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, content in (("train.ppm", "training"), ("test.ppm", "evaluation")):
            (self.root / name).write_text(content)
        self.rows = [
            ["train", "train.ppm", "plate", "train", "capture_a", sha256(self.root / "train.ppm")],
            ["test", "test.ppm", "plate", "test", "capture_b", sha256(self.root / "test.ppm")],
        ]
        self.metadata = {"schema_version": 1, "name": "fixture", "purpose": "holdout",
                         "manifest": "samples.csv", "annotation_status": "owner_verified",
                         "training": {"kind": "images"}, "collection_protocol": "Independent fixture inputs."}

    def load(self):
        with (self.root / "samples.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SAMPLE_FIELDS)
            writer.writerows(self.rows)
        path = self.root / "dataset.json"
        path.write_text(json.dumps(self.metadata))
        return load_dataset(path)

    def test_verified_manifest_loads(self):
        dataset = self.load()
        self.assertEqual([sample.label for sample in dataset.samples], ["plate", "plate"])

    def test_unreviewed_labels_are_rejected(self):
        self.metadata["annotation_status"] = "pending"
        with self.assertRaisesRegex(ValueError, "Confirm image labels"):
            self.load()

    def test_changed_image_is_rejected(self):
        (self.root / "test.ppm").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load()

    def test_same_content_under_different_names_is_rejected(self):
        (self.root / "test.ppm").write_bytes((self.root / "train.ppm").read_bytes())
        self.rows[1][-1] = self.rows[0][-1]
        with self.assertRaisesRegex(ValueError, "Duplicate image content"):
            self.load()

    def test_capture_group_leakage_is_rejected(self):
        self.rows[1][4] = self.rows[0][4]
        with self.assertRaisesRegex(ValueError, "capture groups overlap"):
            self.load()

    def test_case_alias_sample_ids_are_rejected_on_every_platform(self):
        self.rows[0][0] = "capture"
        self.rows[1][0] = "CAPTURE"
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            self.load()

    def test_group_whitespace_cannot_bypass_split_checks(self):
        for value in ("capture_a ", "unknown "):
            with self.subTest(value=value):
                self.rows[1][4] = value
                with self.assertRaisesRegex(ValueError, "capture group"):
                    self.load()

    def test_unknown_group_cannot_be_a_holdout(self):
        self.rows[1][4] = "unknown"
        with self.assertRaisesRegex(ValueError, "known capture groups"):
            self.load()

    def test_legacy_features_cannot_claim_holdout(self):
        self.metadata["training"] = {"kind": "legacy_features"}
        with self.assertRaisesRegex(ValueError, "only for smoke"):
            self.load()

    def test_six_dimensional_or_nonfinite_features_are_rejected(self):
        path = self.root / "features.csv"
        path.write_text("label,f0,f1,f2,f3,f4,f5\nplate,1,2,3,4,5,6\n")
        with self.assertRaisesRegex(ValueError, "exactly"):
            load_legacy_features(path)
        path.write_text("label,f0,f1,f2,f3\nplate,1,2,nan,4\n")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            load_legacy_features(path)

    def test_failure_rows_remain_in_accuracy_denominator(self):
        rows = [
            {"true_label": "plate", "predicted_label": "plate", "correct": "1", "status": "ok", "latency_ms": "10", "detection_count": "1"},
            {"true_label": "plate", "predicted_label": "NoDetection", "correct": "0", "status": "no_detection", "latency_ms": "20", "detection_count": "0"},
            {"true_label": "plate", "predicted_label": "UnreadableImage", "correct": "0", "status": "unreadable_image", "latency_ms": "999", "detection_count": "0"},
        ]
        metrics = summarize(rows, "smoke")
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["pipeline_latency_ms"]["samples"], 2)
        self.assertEqual(metrics["pipeline_latency_ms"]["p50"], 15)
        self.assertAlmostEqual(metrics["pipeline_latency_ms"]["p95"], 19.5)
        self.assertFalse(metrics["independent_split_declared"])


if __name__ == "__main__":
    unittest.main()
