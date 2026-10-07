#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

from evaluation import summarize


RESULT_FIELDS = {"frame_id", "true_label", "predicted_label", "score", "latency_ms", "correct", "status", "detection_count"}


def read_results(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = RESULT_FIELDS - set(fields)
        if missing:
            raise ValueError("Evaluation CSV is missing required fields: " + ",".join(sorted(missing)))
        if len(fields) != len(set(fields)):
            raise ValueError("Evaluation CSV contains duplicate column names.")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Evaluation CSV contains a malformed row.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize evaluation CSV into a compact metrics JSON.")
    parser.add_argument("input", type=Path, nargs="?", default=Path("data/db/eval_results.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--purpose", choices=["smoke", "holdout"], default="smoke",
                        help="Declared experiment purpose; this command does not verify split independence.")
    args = parser.parse_args()
    try:
        metrics = summarize(read_results(args.input), args.purpose)
        serialized = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
            print(f"Wrote metrics to {args.output}")
        else:
            print(serialized, end="")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"Metrics export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
