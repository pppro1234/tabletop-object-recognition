# Tabletop Object Recognition

A C++17/OpenCV tabletop object recognition system with a modular pipeline, automated regression checks, and reproducible offline experiments.

The main pipeline separates preprocessing, segmentation, morphology, region analysis, four-dimensional shape features, nearest-neighbor classification, centroid tracking, rendering, and SQLite storage. Three command-line tools share that implementation:

| Tool | Purpose |
|---|---|
| `tabletop_recognition_live` | Interactive webcam recognition and visualization |
| `tabletop_recognition_train` | Save labeled shape features from camera frames or image files |
| `tabletop_recognition_eval` | Evaluate explicitly labeled images and preserve every outcome |

## Build, test, and reproduce

Requirements: a C++17 compiler, CMake 3.21+, OpenCV development libraries, SQLite3 development libraries, and Python 3.10+. The evaluation scripts use the Python standard library.

Run from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 scripts/run_eval.py --out artifacts/evaluations/real-smoke-001
```

Read `artifacts/evaluations/real-smoke-001/report.md` for the results. Each experiment gets a new database, frozen inputs, configuration evidence, hashes, per-image predictions, summary metrics, and failure overlays. Existing output directories are rejected; use a new name for another run, or omit `--out` for an automatically generated name.

The default dataset contains **five real images with owner-confirmed labels**. It uses a fixed snapshot of the original four-dimensional training features. The training samples' image provenance and independence from these five images are unverified, so this is a **smoke check, not a held-out accuracy benchmark**. See the [dataset notes](data/evaluation/real_smoke/README.md).

To validate the dataset without executing recognition:

```sh
python3 scripts/run_eval.py --validate-only
```

See [evaluation and metric definitions](docs/evaluation.md) and [build/test details](docs/build-and-test.md).

## Evaluation contract

`tabletop_recognition_eval` requires a valid `image,label` CSV supplied through `--manifest` or the configuration. It never guesses truth labels from filenames. It evaluates only listed images, retains missing/unreadable images as failure rows, and returns a nonzero exit code for processing errors.

The current image-level evaluation assumes one labeled object per image and scores the largest retained region. `NoDetection` is a failed detection; `Unknown` means a detected object was rejected by the classifier. They are separate outcomes. Results do not measure multi-object detection or tracking quality.

CTest covers unit, integration, regression, and performance checks, including manifest validation and experiment isolation. Synthetic fixtures catch software regressions; their accuracy and timing do not establish real-world recognition performance.

## Interactive tools

After building:

```sh
./build/tabletop_recognition_live --config configs/webcam.yaml
./build/tabletop_recognition_train --config configs/default.yaml --label mug --source webcam
```

In webcam training, press `s` to save the primary detection and `q` to quit. Training stores feature vectors and sample images; it does not train a neural network. Configuration files control segmentation, region filtering, classifier thresholds, storage, and observability.

## Sample, save, evaluate, inspect failures

For a labeled image, use `tabletop_recognition_train` to save its feature row and sample image. Pass separate output paths when trying new material so the existing database and samples stay intact:

```sh
./build/tabletop_recognition_train --config configs/default.yaml --source /path/to/train-photo.jpg \
  --label mug --database /path/to/training.sqlite3 \
  --sample-dir /path/to/saved-samples --artifacts /path/to/training-runs
```

Saving writes the image before its database row and removes this attempt's new image after a handled failure; an abrupt process or system termination between those steps can still leave an orphan image.

For a repeatable evaluation, record owner-reviewed images in a CSV with `sample_id,image,label,split,group_id`, then run `scripts/prepare_dataset.py` to add image hashes. Create a dataset descriptor with `training.kind: images` as described in [evaluation setup](docs/evaluation.md#dataset-descriptor-and-sample-metadata). The runner trains a fresh, isolated database from its `train` rows and evaluates its `test` rows; the default five-image smoke run above instead imports fixed legacy features and does not read the manual training database.

```sh
python3 scripts/prepare_dataset.py /path/to/reviewed_annotations.csv \
  --output /path/to/new-set/samples.csv
python3 scripts/run_eval.py --dataset /path/to/new-set/dataset.json \
  --out artifacts/evaluations/new-set-001
```

Inspect `report.md` and `results/eval_results.csv` in the new output directory for every prediction and status, `artifacts/failures/` for available failure overlays, and `samples.csv` plus `provenance.json` for frozen inputs and source/configuration hashes. Missing or unreadable images still have result rows but cannot have overlays. Use a new output directory for each run.

The included five photos only check that this workflow runs. To assess recognition on new data, collect labeled training and test photos from separate sessions, including varied poses and lighting, unfamiliar objects, and empty scenes. Keep images from the same recording or physical capture group in one split, confirm labels and source records before preparing the manifest, and avoid tuning repeatedly against the held-out test set. More detail is in [evaluation setup](docs/evaluation.md).

## Scope and status

- The main build uses `shape4d` features and nearest-neighbor classification. ResNet embedding extraction and feature fusion are not part of the current pipeline.
- `service/api/` is a scaffold: `/predict` currently returns HTTP 501. API completion and production deployment are outside the current deliverables.
- `.github/workflows/ci.yml` defines a GitHub Actions build/test workflow. There is currently no configured remote, so a hosted workflow run has not been verified. Local CTest results and remote CI execution are separate evidence.

## Layout

```text
include/vision/        Shared types and module interfaces
src/core/             Pipeline modules and SQLite storage
src/app/              Live, training, and evaluation tools
configs/              Runtime configuration
scripts/              Dataset validation and experiment runner
data/evaluation/      Versioned evaluation inputs and provenance notes
tests/                Unit, integration, regression, performance checks
docs/                 Architecture, build, and evaluation documentation
service/api/          Unfinished API scaffold
```
