This directory holds the golden fixtures for regression testing.

Files:
- `golden/train/`: synthetic training images used to seed the classifier in deterministic tests
- `golden/eval/`: fixed evaluation inputs
- `golden/expected_eval.csv`: expected label, centroid, angle, and bounding box for each evaluation image
- `golden/manifest.csv`: dataset manifest for dataset-style evaluation runs
- `generate_golden_dataset.py`: regenerates the full golden set from deterministic geometry parameters

Why this exists:
- regression tests need stable inputs and stable expected outputs
- new code changes should not silently break classification, pose estimation, or localization
- the golden set gives the project a repeatable quality gate that can run locally and in CI

Regenerate the fixtures with:

```bash
python3 tests/regression/generate_golden_dataset.py
```

The generator clears old generated `*.ppm` and `*.csv` files first so stale fixtures do not survive between revisions.
