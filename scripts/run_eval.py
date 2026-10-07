#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from evaluation import load_dataset, run_experiment


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run an isolated, manifest-driven recognition experiment.")
    parser.add_argument("--dataset", type=Path, default=root / "data/evaluation/real_smoke/dataset.json")
    parser.add_argument("--binary", type=Path, default=root / "build/tabletop_recognition_eval")
    parser.add_argument("--train-binary", type=Path, default=root / "build/tabletop_recognition_train")
    parser.add_argument("--config", type=Path, default=root / "configs/eval.yaml")
    parser.add_argument("--out", type=Path, help="New output directory; existing directories are never overwritten.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.dataset)
        if args.validate_only:
            print(f"Validated {dataset.name}: {len(dataset.samples)} images; purpose={dataset.purpose}")
            return 0
        output, success = run_experiment(dataset, args.binary, args.train_binary, args.config, args.out, root)
        print(f"Report: {output / 'report.md'}")
        return 0 if success else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
