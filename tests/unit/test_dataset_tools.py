#!/usr/bin/env python3
"""Check explicit annotation preparation and evaluation CSV export contracts."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from evaluation import SAMPLE_FIELDS, sha256
from export_metrics import read_results
from prepare_dataset import ANNOTATION_FIELDS, prepare_manifest


class DatasetToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dataset-tools-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotation_dir = self.root / "annotations"
        self.annotation_dir.mkdir()
        self.image = self.annotation_dir / 'camera_01,"source".jpg'
        self.image.write_bytes(b"image content is fingerprinted without inferring labels")
        self.second_image = self.annotation_dir / "other.jpg"
        self.second_image.write_bytes(b"a distinct capture")
        self.annotations = self.annotation_dir / "reviewed.csv"
        self.output = self.root / "dataset" / "samples.csv"
        self.rows = [{"sample_id": "sample_1", "image": self.image.name,
                      "label": 'part, "left"', "split": "test", "group_id": "capture_1"}]

    def write_annotations(self):
        with self.annotations.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def prepare(self):
        self.write_annotations()
        return prepare_manifest(self.annotations, self.output)

    def test_explicit_label_and_relative_path_survive_preparation(self):
        self.assertEqual(self.prepare(), 1)
        with self.output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, SAMPLE_FIELDS)
            records = list(reader)
        self.assertEqual(records[0]["label"], 'part, "left"')
        self.assertEqual(records[0]["sha256"], sha256(self.image))
        self.assertFalse(Path(records[0]["image"]).is_absolute())
        self.assertEqual((self.output.parent / records[0]["image"]).resolve(), self.image.resolve())
        self.assertFalse((self.output.parent / "dataset.json").exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.parent.mkdir()
        self.output.write_text("preserve me", encoding="utf-8")
        self.write_annotations()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            prepare_manifest(self.annotations, self.output)
        self.assertEqual(self.output.read_text(), "preserve me")

    def test_invalid_explicit_annotations_are_rejected(self):
        original = self.rows[0].copy()
        for field, value, message in (("label", "", "label"), ("split", "validation", "split"),
                                      ("group_id", "", "group_id"), ("sample_id", "not a token", "sample_id")):
            with self.subTest(field=field):
                self.rows = [{**original, field: value}]
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_duplicate_ids_paths_and_image_content_are_rejected(self):
        original = self.rows[0].copy()
        cases = [
            ({**original, "image": self.second_image.name}, "sample_id"),
            ({**original, "sample_id": "SAMPLE_1", "image": self.second_image.name}, "sample_id"),
            ({**original, "sample_id": "sample_2"}, "duplicate image"),
        ]
        for second, message in cases:
            with self.subTest(message=message):
                self.rows = [original, second]
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()
                self.assertFalse(self.output.exists())
        self.second_image.write_bytes(self.image.read_bytes())
        self.rows = [original, {**original, "sample_id": "sample_2", "image": self.second_image.name}]
        with self.assertRaisesRegex(ValueError, "Duplicate image content"):
            self.prepare()

    def test_missing_file_and_capture_group_leakage_are_rejected(self):
        original = self.rows[0].copy()
        self.rows[0]["image"] = "missing.jpg"
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.rows = [original, {**original, "sample_id": "train_1", "image": self.second_image.name, "split": "train"}]
        with self.assertRaisesRegex(ValueError, "both train and test"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_prepare_cli_does_not_depend_on_working_directory(self):
        self.write_annotations()
        result = subprocess.run([sys.executable, "-B", str(SCRIPTS / "prepare_dataset.py"),
                                 str(self.annotations), "--output", str(self.output)],
                                cwd=self.root, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no owner review status was assigned", result.stdout)
        self.assertTrue(self.output.is_file())


class MetricsToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="metrics-tools-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "results.csv"
        self.fields = ["frame_id", "true_label", "predicted_label", "score", "latency_ms", "correct",
                       "status", "detection_count"]
        self.rows = [
            {"frame_id": "one.jpg", "true_label": 'part, "left"', "predicted_label": 'part, "left"',
             "score": "0.75", "latency_ms": "10", "correct": "1", "status": "ok", "detection_count": "1"},
            {"frame_id": "empty.jpg", "true_label": "Unknown", "predicted_label": "NoDetection",
             "score": "0", "latency_ms": "20", "correct": "0", "status": "no_detection", "detection_count": "0"},
        ]

    def write_results(self):
        with self.input.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def export(self, *arguments):
        self.write_results()
        return subprocess.run([sys.executable, "-B", str(SCRIPTS / "export_metrics.py"), str(self.input), *arguments],
                              cwd=self.root, text=True, capture_output=True, check=False)

    def test_export_defaults_to_smoke_and_counts_detection_failure(self):
        result = self.export()
        self.assertEqual(result.returncode, 0, result.stderr)
        metrics = json.loads(result.stdout)
        self.assertEqual(metrics["purpose"], "smoke")
        self.assertFalse(metrics["independent_split_declared"])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["status_counts"], {"no_detection": 1, "ok": 1})
        self.assertEqual(metrics["per_class"]['part, "left"']["correct"], 1)

    def test_export_accepts_explicit_purpose_and_output(self):
        output = self.root / "report" / "summary.json"
        result = self.export("--purpose", "holdout", "--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["purpose"], "holdout")

    def test_old_csv_without_status_fields_is_rejected(self):
        self.fields.remove("status")
        self.fields.remove("detection_count")
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required fields", result.stderr)
        self.assertIn("status", result.stderr)
        self.assertIn("detection_count", result.stderr)

    def test_forged_correct_flag_cannot_inflate_accuracy(self):
        self.rows[1]["correct"] = "1"
        output = self.root / "summary.json"
        result = self.export("--output", str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Metrics export failed", result.stderr)
        self.assertFalse(output.exists())

    def test_extra_csv_values_are_rejected(self):
        self.write_results()
        with self.input.open("a", encoding="utf-8") as handle:
            handle.write("a,b,c,d,e,f,g,h,extra\n")
        with self.assertRaisesRegex(ValueError, "malformed"):
            read_results(self.input)


if __name__ == "__main__":
    unittest.main()
