"""Validate datasets and preserve the inputs and outputs of each experiment."""

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import subprocess
import uuid


RESERVED_LABELS = {"NoDetection", "MissingImage", "UnreadableImage", "PipelineError"}
SAMPLE_FIELDS = ["sample_id", "image", "label", "split", "group_id", "sha256"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def valid_label(value: str) -> bool:
    return bool(value and value == value.strip() and value not in RESERVED_LABELS
                and not any(ord(char) < 32 for char in value))


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: Path
    label: str
    split: str
    group_id: str
    sha256: str


@dataclass(frozen=True)
class Dataset:
    path: Path
    name: str
    purpose: str
    metadata: dict
    samples: tuple[Sample, ...]
    training_features: Path | None


def load_dataset(path: Path) -> Dataset:
    path = path.resolve(strict=True)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Dataset schema_version must be 1.")
    name = metadata.get("name", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Dataset name must contain only letters, digits, underscores, and hyphens.")
    purpose = metadata.get("purpose")
    if purpose not in {"smoke", "holdout"}:
        raise ValueError("Dataset purpose must be smoke or holdout.")
    if metadata.get("annotation_status") != "owner_verified":
        raise ValueError("Confirm image labels with the data owner before evaluating them.")
    manifest = path.parent / metadata["manifest"]
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SAMPLE_FIELDS:
            raise ValueError(f"Dataset manifest header must be {','.join(SAMPLE_FIELDS)}")
        rows = list(reader)
    samples = []
    seen_ids, seen_paths, seen_hashes = set(), set(), set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed dataset manifest row.")
        sample_id = row["sample_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id) or sample_id.casefold() in seen_ids:
            raise ValueError(f"Invalid or duplicate sample_id: {sample_id}")
        image = (manifest.parent / row["image"]).resolve(strict=True)
        if not image.is_file() or image in seen_paths:
            raise ValueError(f"Invalid or duplicate image: {image}")
        actual_hash = sha256(image)
        if actual_hash != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {sample_id}")
        if actual_hash in seen_hashes:
            raise ValueError(f"Duplicate image content: {sample_id}")
        if not valid_label(row["label"]):
            raise ValueError(f"Invalid label: {sample_id}")
        group = row["group_id"]
        if (row["split"] not in {"train", "test"} or not group or group != group.strip()
                or any(ord(char) < 32 for char in group)):
            raise ValueError(f"Missing split or capture group: {sample_id}")
        samples.append(Sample(sample_id, image, row["label"], row["split"], row["group_id"], actual_hash))
        seen_ids.add(sample_id.casefold())
        seen_paths.add(image)
        seen_hashes.add(actual_hash)
    if not any(sample.split == "test" for sample in samples):
        raise ValueError("At least one test image is required.")
    training = metadata.get("training", {})
    training_features = None
    if training.get("kind") == "legacy_features":
        if purpose != "smoke" or any(sample.split == "train" for sample in samples):
            raise ValueError("Legacy features are supported only for smoke datasets without train images.")
        training_features = (path.parent / training["path"]).resolve(strict=True)
        if sha256(training_features) != training.get("sha256"):
            raise ValueError("Training feature checksum mismatch.")
        load_legacy_features(training_features)
    elif training.get("kind") == "images":
        train = [sample for sample in samples if sample.split == "train"]
        test = [sample for sample in samples if sample.split == "test"]
        if not train:
            raise ValueError("Image-based training requires at least one train image.")
        if any(sample.label == "Unknown" for sample in train):
            raise ValueError("Unknown is reserved for unseen test objects, not training samples.")
        train_groups = {sample.group_id for sample in train}
        if train_groups & {sample.group_id for sample in test}:
            raise ValueError("Training and test capture groups overlap.")
        if purpose == "holdout" and any(sample.group_id.lower() in {"unknown", "unverified"} for sample in samples):
            raise ValueError("Holdout data requires known capture groups.")
    else:
        raise ValueError("Training kind must be images or legacy_features.")
    if purpose == "holdout" and not metadata.get("collection_protocol"):
        raise ValueError("Holdout datasets must document their collection protocol.")
    return Dataset(path, name, purpose, metadata, tuple(samples), training_features)


def load_legacy_features(path: Path) -> list[tuple[str, list[float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["label", "f0", "f1", "f2", "f3"]:
            raise ValueError("Legacy shape4d features require exactly label,f0,f1,f2,f3.")
        result = []
        for row in reader:
            if None in row or not valid_label(row["label"]) or row["label"] == "Unknown":
                raise ValueError("Malformed legacy training row.")
            vector = [float(row[f"f{index}"]) for index in range(4)]
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Non-finite training feature.")
            result.append((row["label"], vector))
        if not result:
            raise ValueError("Training feature snapshot is empty.")
        return result


def seed_legacy_database(database: Path, features: Path) -> None:
    # These vectors already contain signed-log Hu moments; do not transform them again.
    records = load_legacy_features(features)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE object_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                           "label TEXT NOT NULL, feature_vector TEXT NOT NULL, embedding_vector TEXT, "
                           "image_path TEXT, created_at TEXT NOT NULL)")
        connection.executemany("INSERT INTO object_samples(label, feature_vector, embedding_vector, image_path, created_at) "
                               "VALUES (?, ?, '', '', '')",
                               [(label, ",".join(format(value, ".17g") for value in vector)) for label, vector in records])


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[dict], purpose: str) -> dict:
    if not rows:
        raise ValueError("Evaluator produced no predictions.")
    failure_labels = {"no_detection": "NoDetection", "missing_image": "MissingImage",
                      "unreadable_image": "UnreadableImage", "pipeline_error": "PipelineError"}
    for row in rows:
        status = row["status"]
        count = int(row["detection_count"])
        detected = status in {"ok", "misclassified"}
        if status not in {"ok", "misclassified", *failure_labels} or count < 0:
            raise ValueError("Invalid prediction status or detection count.")
        if detected != (count > 0):
            raise ValueError("Prediction status disagrees with detection count.")
        if status in failure_labels and row["predicted_label"] != failure_labels[status]:
            raise ValueError("Failure status must use its explicit outcome label.")
        expected_correct = detected and row["true_label"] == row["predicted_label"]
        if (status == "ok") != expected_correct or row["correct"] != str(int(expected_correct)):
            raise ValueError("Prediction correct field disagrees with labels or status.")
    correct = sum(row["correct"] == "1" for row in rows)
    pipeline_rows = [row for row in rows if row["status"] in {"ok", "misclassified", "no_detection"}]
    latencies = [float(row["latency_ms"]) for row in pipeline_rows]
    if any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("Invalid pipeline latency in evaluation output.")
    per_class = {}
    for label in sorted({row["true_label"] for row in rows}):
        actual = [row for row in rows if row["true_label"] == label]
        predicted = [row for row in rows if row["predicted_label"] == label]
        true_positive = sum(row["correct"] == "1" for row in actual)
        per_class[label] = {"support": len(actual), "correct": true_positive,
                            "recall": true_positive / len(actual),
                            "precision": true_positive / len(predicted) if predicted else 0.0}
    return {
        "purpose": purpose, "independent_split_declared": purpose == "holdout",
        "total": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "per_class": per_class,
        "pipeline_latency_ms": {"samples": len(latencies), "mean": sum(latencies) / len(latencies) if latencies else None,
                                "p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "timing_scope": "Pipeline including rendering; excludes image decoding, logging, and failure image writes. "
                        "No warmup; missing, unreadable, and exception rows excluded from latency percentiles.",
    }


def invoke(command: list[str], log: Path, cwd: Path) -> int:
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def source_identity(root: Path) -> dict:
    paths = [root / "CMakeLists.txt"]
    for folder in ("src/app", "src/core", "include", "scripts", "configs"):
        paths.extend(path for path in (root / folder).rglob("*") if path.suffix in {".cpp", ".hpp", ".py", ".yaml"})
    result = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None,
            "working_tree_dirty": bool(status.stdout) if status.returncode == 0 else None,
            "files": {str(path.relative_to(root)): sha256(path) for path in sorted(paths) if path.is_file()}}


def run_experiment(dataset: Dataset, binary: Path, train_binary: Path, config: Path,
                   output: Path | None, root: Path) -> tuple[Path, bool]:
    binary, config = binary.resolve(strict=True), config.resolve(strict=True)
    if not os.access(binary, os.X_OK):
        raise ValueError(f"Evaluator is not executable: {binary}")
    if dataset.training_features is None:
        train_binary = train_binary.resolve(strict=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (output or root / "artifacts/evaluations" / f"{dataset.name}_{timestamp}_{uuid.uuid4().hex[:8]}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    database = output / "vision.sqlite3"
    frozen = []
    for sample in dataset.samples:
        target = inputs / (sample.sample_id + sample.image.suffix.lower())
        # Exclusive creation prevents accidental aliases from replacing frozen inputs.
        with sample.image.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        if sha256(target) != sample.sha256:
            raise ValueError(f"Image changed while snapshotting: {sample.sample_id}")
        frozen.append(Sample(sample.sample_id, target, sample.label, sample.split, sample.group_id, sample.sha256))
    config_snapshot = output / "config_source.yaml"
    shutil.copyfile(config, config_snapshot)
    shutil.copyfile(dataset.path, output / "dataset_original.json")
    frozen_metadata = json.loads(json.dumps(dataset.metadata))
    frozen_metadata["manifest"] = "samples.csv"
    if dataset.training_features is not None:
        frozen_metadata["training"]["path"] = "training_features.csv"
    write_json(output / "dataset_source.json", frozen_metadata)
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLE_FIELDS)
        for sample in frozen:
            writer.writerow([sample.sample_id, str(sample.image.relative_to(output)), sample.label,
                             sample.split, sample.group_id, sample.sha256])
    provenance = {"dataset": dataset.name, "purpose": dataset.purpose,
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "dataset_descriptor_sha256": sha256(dataset.path), "config_sha256": sha256(config_snapshot),
                  "evaluator_sha256": sha256(binary), "source": source_identity(root),
                  "environment": {"os": platform.platform(), "architecture": platform.machine(),
                                  "cpu_count": os.cpu_count(), "python": platform.python_version()}, "commands": []}
    common = ["--config", str(config_snapshot), "--database", str(database), "--sample-dir", str(output / "training_crops")]
    if dataset.training_features is not None:
        features = output / "training_features.csv"
        shutil.copyfile(dataset.training_features, features)
        if sha256(features) != dataset.metadata["training"]["sha256"]:
            raise ValueError("Training features changed while snapshotting.")
        seed_legacy_database(database, features)
        provenance["training_features_sha256"] = sha256(features)
    else:
        provenance["trainer_sha256"] = sha256(train_binary)
        for sample in frozen:
            if sample.split != "train":
                continue
            command = [str(train_binary), *common, "--artifacts", str(output / "training_artifacts" / sample.sample_id),
                       "--source", str(sample.image), "--label", sample.label]
            provenance["commands"].append(command)
            write_json(output / "provenance.json", provenance)
            if invoke(command, output / f"train_{sample.sample_id}.log", root) != 0:
                raise RuntimeError(f"Training failed for {sample.sample_id}; see {output}")
    test_samples = [sample for sample in frozen if sample.split == "test"]
    manifest = output / "eval_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label"])
        writer.writerows((str(sample.image), sample.label) for sample in test_samples)
    command = [str(binary), *common, "--artifacts", str(output / "artifacts"),
               "--manifest", str(manifest), "--out", str(output / "results")]
    provenance["commands"].append(command)
    write_json(output / "provenance.json", provenance)
    returncode = invoke(command, output / "eval.log", root)
    predictions = output / "results/eval_results.csv"
    if not predictions.is_file():
        raise RuntimeError(f"Evaluator produced no prediction file; see {output / 'eval.log'}")
    with predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {str(sample.image): sample for sample in test_samples}
    if len(rows) != len(expected) or {row["frame_id"] for row in rows} != set(expected):
        raise RuntimeError("Evaluator did not return exactly one row per test image.")
    for row in rows:
        if row["true_label"] != expected[row["frame_id"]].label:
            raise RuntimeError("Evaluator ground truth does not match the frozen manifest.")
    summary = summarize(rows, dataset.purpose)
    summary["dataset"] = dataset.name
    summary["evaluator_exit_code"] = returncode
    summary["limitations"] = dataset.metadata.get("limitations", [])
    write_json(output / "summary.json", summary)
    provenance["completed_at"] = datetime.now(timezone.utc).isoformat()
    provenance["evaluator_exit_code"] = returncode
    write_json(output / "provenance.json", provenance)
    lines = [f"# Evaluation: {dataset.name}", "", f"Purpose: **{dataset.purpose}**.", ""]
    if dataset.purpose == "smoke":
        lines += ["This is a fixed-image smoke check. Its accuracy is descriptive only and must not be "
                  "presented as held-out or generalization performance.", ""]
    lines += [f"Correct: **{summary['correct']}/{summary['total']}** ({summary['accuracy']:.1%}). "
              "Missing detections and processing failures remain in the denominator.", "",
              f"Evaluator exit code: {returncode}.", "", "## Predictions", "",
              "| Sample | True label | Prediction | Status | Detections | Measured stage latency (ms) |",
              "|---|---|---|---|---:|---:|"]
    for row in rows:
        sample = expected[row["frame_id"]]
        values = [sample.sample_id, row["true_label"], row["predicted_label"], row["status"],
                  row["detection_count"], f"{float(row['latency_ms']):.3f}"]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    latency = summary["pipeline_latency_ms"]
    lines += ["", "## Timing", "", summary["timing_scope"], "",
              f"Pipeline latency: mean={latency['mean']}, P50={latency['p50']}, P95={latency['p95']} ms.", "",
              "These measurements describe this machine and input set, not live-camera throughput.", "",
              "## Limitations", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines += ["", "## Artifacts", "", "- `results/eval_results.csv`: per-image results.",
              "- `results/confusion_matrix.csv`: all outcomes, including detection failures.",
              "- `results/metrics.json`: native pipeline and end-to-end timing metrics.",
              "- `artifacts/failures/`: failed-image overlays.",
              "- `artifacts/runs/`: resolved configuration and frame logs.",
              "- `inputs/`, `samples.csv`, `training_features.csv` (legacy mode): frozen experiment data.",
              "- `provenance.json`: source, binary, configuration, and data fingerprints.",
              "- `vision.sqlite3`: isolated samples, run, and predictions.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output, returncode == 0
