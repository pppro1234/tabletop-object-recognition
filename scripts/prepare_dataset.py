#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import re
import sys

from evaluation import SAMPLE_FIELDS, sha256, valid_label


ANNOTATION_FIELDS = ["sample_id", "image", "label", "split", "group_id"]


def prepare_manifest(annotations: Path, output: Path) -> int:
    """Fingerprint explicitly annotated images without inferring labels or review status."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    annotations = annotations.resolve(strict=True)
    output = output.absolute()
    with annotations.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ANNOTATION_FIELDS:
            raise ValueError(f"Annotation CSV header must be {','.join(ANNOTATION_FIELDS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Annotation CSV is empty.")

    result = []
    seen_ids, seen_images, seen_hashes = set(), set(), set()
    group_splits = {}
    for number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed annotation row {number}.")
        sample_id = row["sample_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id) or sample_id.casefold() in seen_ids:
            raise ValueError(f"Invalid or duplicate sample_id at row {number}: {sample_id}")
        if not valid_label(row["label"]):
            raise ValueError(f"Invalid or empty label at row {number}.")
        if row["split"] not in {"train", "test"}:
            raise ValueError(f"Invalid split at row {number}: use train or test.")
        group = row["group_id"]
        if not group or group != group.strip() or any(ord(char) < 32 for char in group):
            raise ValueError(f"Invalid or empty group_id at row {number}.")
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError(f"Capture group appears in both train and test: {group}")
        if not row["image"].strip():
            raise ValueError(f"Missing image path at row {number}.")
        image = (annotations.parent / row["image"]).resolve(strict=True)
        if not image.is_file() or image in seen_images:
            raise ValueError(f"Invalid or duplicate image at row {number}: {image}")
        digest = sha256(image)
        if digest in seen_hashes:
            raise ValueError(f"Duplicate image content at row {number}: {image}")
        relative_image = Path(os.path.relpath(image, output.parent.resolve())).as_posix()
        result.append({**row, "image": relative_image, "sha256": digest})
        seen_ids.add(sample_id.casefold())
        seen_images.add(image)
        seen_hashes.add(digest)
        group_splits[group] = row["split"]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(result)
    return len(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fingerprint images from an explicitly reviewed annotation CSV.")
    parser.add_argument("annotations", type=Path,
                        help="CSV with sample_id,image,label,split,group_id; image paths are relative to this CSV.")
    parser.add_argument("--output", type=Path, required=True, help="New samples CSV; existing files are never overwritten.")
    args = parser.parse_args()
    try:
        count = prepare_manifest(args.annotations, args.output)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {count} samples to {args.output}.")
    print("Create and confirm the dataset descriptor separately; no owner review status was assigned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
