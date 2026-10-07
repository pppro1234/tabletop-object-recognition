#!/usr/bin/env python3
"""Exercise the evaluator's manifest contract through its public CLI."""

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest


BINARY = None


class EvaluationCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vision-eval-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.output = self.root / "results" / "evaluation.csv"
        self.database = self.root / "evaluation.sqlite3"
        self.artifacts = self.root / "artifacts"
        self.config = self.root / "eval.yaml"
        self.config.write_text(
            '%YAML:1.0\n'
            'run_name: "manifest_contract"\n'
            'input:\n  image_dir: "unused_directory"\n  image_glob: "*.jpg"\n'
            'region:\n  min_area: 20\n  max_regions: 1\n  reject_boundary: 1\n'
            'rendering:\n  draw_debug_views: 0\n  draw_labels: 0\n'
            'evaluation:\n  manifest_path: "manifest.csv"\n'
            'observability:\n  save_failure_samples: 1\n  save_config_snapshot: 1\n  log_per_frame: 1\n',
            encoding="utf-8",
        )

    def image(self, relative, object_present=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = bytearray()
        for y in range(64):
            for x in range(64):
                value = 20 if object_present and 20 <= x < 44 and 16 <= y < 48 else 255
                pixels.extend((value, value, value))
        path.write_bytes(b"P6\n64 64\n255\n" + pixels)
        return path

    def write_manifest(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "label"])
            writer.writerows(rows)

    def run_eval(self, *extra):
        return subprocess.run(
            [str(BINARY), "--config", str(self.config), "--database", str(self.database),
             "--artifacts", str(self.artifacts), "--out", str(self.output), *map(str, extra)],
            cwd=self.root, capture_output=True, text=True, timeout=30,
        )

    def rows(self):
        with self.output.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_missing_manifest_fails_before_creating_run(self):
        result = self.run_eval()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot open evaluation manifest", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.database.exists())
        self.assertFalse(self.artifacts.exists())

    def test_invalid_manifests_are_rejected(self):
        invalid = [
            "", "image,label\n", "filename,label\na.ppm,a\n",
            "image,label,extra\na.ppm,a,1\n", "image,label\na.ppm,\n",
            "image,label\n,a\n", 'image,label\n"a.ppm,a\n',
            'image,label\n"a.ppm"junk,a\n', "image,label\n\n",
            "image,label\na.ppm,a\n./a.ppm,b\n",
            "image,label\na.ppm, Unknown\n", "image,label\na.ppm,Unknown \n",
        ]
        invalid.extend(f"image,label\na.ppm,{label}\n" for label in
                       ("NoDetection", "MissingImage", "UnreadableImage", "PipelineError"))
        for content in invalid:
            with self.subTest(content=content):
                self.manifest.write_text(content, encoding="utf-8")
                result = self.run_eval()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.database.exists())

    def test_only_manifest_images_are_evaluated_with_exact_paths_and_labels(self):
        first = self.image("first/same.ppm")
        second = self.image('second/same,"quoted".ppm')
        self.image("unlisted.ppm")
        self.write_manifest([(first.relative_to(self.root), "truth_one"),
                             (second.relative_to(self.root), 'truth,"two"')])
        result = self.run_eval()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["frame_id"] for row in rows], [str(first.resolve()), str(second.resolve())])
        self.assertEqual([row["true_label"] for row in rows], ["truth_one", 'truth,"two"'])
        snapshots = list(self.artifacts.glob("runs/*/eval_manifest.csv"))
        self.assertEqual(len(snapshots), 1)
        with snapshots[0].open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual([row["image"] for row in saved], [str(first.resolve()), str(second.resolve())])

    def test_same_basename_does_not_overwrite_labels(self):
        self.image("left/same.ppm")
        self.image("right/same.ppm")
        self.write_manifest([("left/same.ppm", "left_label"), ("right/same.ppm", "right_label")])
        result = self.run_eval()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["true_label"] for row in self.rows()], ["left_label", "right_label"])

    def test_no_detection_is_timed_and_is_not_successful_unknown_rejection(self):
        self.image("blank.ppm")
        self.write_manifest([("blank.ppm", "Unknown")])
        result = self.run_eval()
        self.assertEqual(result.returncode, 0, result.stderr)
        row, = self.rows()
        self.assertEqual(row["status"], "no_detection")
        self.assertEqual(row["predicted_label"], "NoDetection")
        self.assertEqual(row["correct"], "0")
        self.assertEqual(row["detection_count"], "0")
        self.assertGreater(float(row["latency_ms"]), 0)
        metrics = json.loads((self.output.parent / "metrics.json").read_text())
        self.assertEqual(metrics["labeled_frames"], 1)
        self.assertEqual(metrics["correct_predictions"], 0)
        with sqlite3.connect(self.database) as connection:
            saved = connection.execute("SELECT status, detection_count, latency_ms FROM predictions").fetchone()
        self.assertEqual(saved[:2], ("no_detection", 0))
        self.assertGreater(saved[2], 0)
        self.assertEqual(len(list(self.artifacts.glob("failures/*.png"))), 1)

    def test_detected_unknown_is_distinct_from_no_detection(self):
        self.image("object.ppm", object_present=True)
        self.write_manifest([("object.ppm", "Unknown")])
        result = self.run_eval()
        self.assertEqual(result.returncode, 0, result.stderr)
        row, = self.rows()
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["predicted_label"], "Unknown")
        self.assertEqual(row["correct"], "1")
        self.assertEqual(row["detection_count"], "1")

    def test_misclassification_records_detection_and_preserves_truth(self):
        self.image("guessed_name.ppm", object_present=True)
        self.write_manifest([("guessed_name.ppm", "authoritative_label")])
        result = self.run_eval()
        self.assertEqual(result.returncode, 0, result.stderr)
        row, = self.rows()
        self.assertEqual(row["status"], "misclassified")
        self.assertEqual(row["true_label"], "authoritative_label")
        self.assertEqual(row["predicted_label"], "Unknown")
        self.assertEqual(row["detection_count"], "1")
        self.assertEqual(row["correct"], "0")

    def test_missing_and_unreadable_images_remain_in_results_and_fail_run(self):
        self.image("valid.ppm")
        (self.root / "broken.ppm").write_text("not an image", encoding="utf-8")
        self.write_manifest([("valid.ppm", "object"), ("missing.ppm", "object"), ("broken.ppm", "object")])
        result = self.run_eval()
        self.assertEqual(result.returncode, 9, result.stderr)
        rows = self.rows()
        self.assertEqual([row["status"] for row in rows], ["no_detection", "missing_image", "unreadable_image"])
        self.assertEqual([row["predicted_label"] for row in rows], ["NoDetection", "MissingImage", "UnreadableImage"])
        self.assertTrue(all(float(row["latency_ms"]) > 0 for row in rows))
        metrics = json.loads((self.output.parent / "metrics.json").read_text())
        self.assertEqual(metrics["processed_frames"], 3)
        self.assertEqual(metrics["labeled_frames"], 3)
        self.assertEqual(metrics["exception_count"], 2)
        self.assertEqual(metrics["completed_pipeline_frames"], 1)
        self.assertEqual(metrics["frames_without_detection"], 1)
        self.assertAlmostEqual(metrics["avg_latency_ms"], float(rows[0]["latency_ms"]), places=5)
        self.assertGreater(metrics["avg_failed_stage_ms"], 0)
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 3)

    def test_dataset_override_is_explicit_and_preserves_relative_subdirectories(self):
        image = self.image("override/nested/image.ppm")
        self.write_manifest([("nested/image.ppm", "object")])
        result = self.run_eval("--dataset", self.root / "override")
        self.assertEqual(result.returncode, 0, result.stderr)
        row, = self.rows()
        self.assertEqual(row["frame_id"], str(image.resolve()))

    def test_symlink_alias_is_a_duplicate(self):
        image = self.image("image.ppm")
        (self.root / "alias.ppm").symlink_to(image)
        self.write_manifest([("image.ppm", "first"), ("alias.ppm", "second")])
        result = self.run_eval()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate image", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
