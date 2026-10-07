#!/usr/bin/env python3
"""Exercise frozen inputs, isolated training, reporting, and replay end to end."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation import SAMPLE_FIELDS, load_dataset, run_experiment, sha256


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="experiment-integration-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_image_training_report_and_frozen_replay(self):
        rows = []
        golden = ROOT / "tests/regression/golden"
        for split, directory in (("train", "train"), ("test", "eval")):
            for source in sorted((golden / directory).glob("*.ppm")):
                target = self.root / source.name
                shutil.copyfile(source, target)
                rows.append([target.stem, target.name, target.stem.split("_")[0], split,
                             f"fixture_{split}", sha256(target)])
        with (self.root / "samples.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SAMPLE_FIELDS)
            writer.writerows(rows)
        descriptor = self.root / "dataset.json"
        descriptor.write_text(json.dumps({"schema_version": 1, "name": "synthetic-contract", "purpose": "smoke",
                                         "manifest": "samples.csv", "annotation_status": "owner_verified",
                                         "training": {"kind": "images"},
                                         "limitations": ["Synthetic contract test; not real-world evidence."]}))
        config = self.root / "config.yaml"
        config.write_text("%YAML:1.0\npreprocess:\n  blur_kernel_size: 3\nsegmentation:\n  sampling_stride: 1\n"
                          "  kmeans_iterations: 20\nmorphology:\n  kernel_size: 3\nregion:\n  min_area: 80\n"
                          "  max_regions: 1\nclassification:\n  unknown_threshold: 100.0\nrendering:\n  draw_debug_views: 0\n")
        dataset = load_dataset(descriptor)
        output, success = run_experiment(dataset, BINARY, TRAIN_BINARY, config, self.root / "first", ROOT)
        self.assertTrue(success)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual((summary["correct"], summary["total"]), (4, 4))
        with sqlite3.connect(output / "vision.sqlite3") as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM object_samples").fetchone()[0], 4)
            self.assertEqual(connection.execute("SELECT count(*) FROM predictions").fetchone()[0], 4)
        with self.assertRaises(FileExistsError):
            run_experiment(dataset, BINARY, TRAIN_BINARY, config, output, ROOT)
        # Replay must use the frozen inputs after the original files have disappeared.
        for source in self.root.glob("*.ppm"):
            source.unlink()
        replay = load_dataset(output / "dataset_source.json")
        repeated, repeated_success = run_experiment(replay, BINARY, TRAIN_BINARY, output / "config_source.yaml",
                                                   self.root / "repeated", ROOT)
        self.assertTrue(repeated_success)
        with (output / "results/eval_results.csv").open(newline="") as first, (repeated / "results/eval_results.csv").open(newline="") as second:
            fields = ("true_label", "predicted_label", "status", "detection_count", "correct")
            a = [tuple(row[key] for key in fields) for row in csv.DictReader(first)]
            b = [tuple(row[key] for key in fields) for row in csv.DictReader(second)]
            self.assertEqual(a, b)

    def test_real_smoke_cli_preserves_sources_and_records_all_five(self):
        descriptor = ROOT / "data/evaluation/real_smoke/dataset.json"
        dataset = load_dataset(descriptor)
        source_pack = self.root / "source-pack"
        for directory in ("metadata", "original-inputs", "reference"):
            (source_pack / directory).mkdir(parents=True)
        metadata = json.loads(descriptor.read_text())
        metadata["manifest"] = "metadata/labels.csv"
        metadata["training"]["path"] = "reference/original-features.csv"
        descriptor = source_pack / "nonstandard.json"
        descriptor.write_text(json.dumps(metadata))
        shutil.copyfile(dataset.training_features, source_pack / metadata["training"]["path"])
        with (source_pack / metadata["manifest"]).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SAMPLE_FIELDS)
            for sample in dataset.samples:
                image_name = sample.sample_id + sample.image.suffix
                shutil.copyfile(sample.image, source_pack / "original-inputs" / image_name)
                writer.writerow([sample.sample_id, f"../original-inputs/{image_name}", sample.label,
                                 sample.split, sample.group_id, sample.sha256])
        original_db = ROOT / "data/db/vision.sqlite3"
        before = sha256(original_db) if original_db.exists() else None
        output = self.root / "real-smoke"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_eval.py"), "--dataset", str(descriptor),
                                 "--binary", str(BINARY), "--out", str(output)], cwd=self.root,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["total"], 5)
        self.assertEqual(sum(summary["status_counts"].values()), 5)
        self.assertFalse(summary["independent_split_declared"])
        self.assertIn("generalization", (output / "report.md").read_text())
        if before is not None:
            self.assertEqual(sha256(original_db), before)
        for sample in dataset.samples:
            self.assertEqual(sha256(sample.image), sample.sha256)
        with sqlite3.connect(output / "vision.sqlite3") as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM object_samples").fetchone()[0], 5)
            self.assertEqual(connection.execute("SELECT count(*) FROM predictions").fetchone()[0], 5)
        with (output / "results/eval_results.csv").open(newline="") as handle:
            failures = sum(row["status"] != "ok" for row in csv.DictReader(handle))
        self.assertEqual(len(list((output / "artifacts/failures").glob("*.png"))), failures)
        shutil.rmtree(source_pack)
        replay = load_dataset(output / "dataset_source.json")
        repeated, success = run_experiment(replay, BINARY, TRAIN_BINARY, output / "config_source.yaml",
                                           self.root / "legacy-replay", ROOT)
        self.assertTrue(success)
        repeated_summary = json.loads((repeated / "summary.json").read_text())
        self.assertEqual(repeated_summary["correct"], summary["correct"])
        self.assertEqual(repeated_summary["status_counts"], summary["status_counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--train-binary", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    TRAIN_BINARY = args.train_binary.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
