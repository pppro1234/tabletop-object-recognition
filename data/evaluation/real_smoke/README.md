# Real-image smoke dataset

This pack contains five existing real single-object images. The owner confirmed their class labels, recorded as `annotation_status: owner_verified` in `dataset.json`.

Its purpose is to exercise the refactored pipeline on fixed real inputs while preserving labels, configuration, failures, and experiment identity. It is not an independent test set or a generalization benchmark.

The confirmed mapping is:

| Sample | Class identifier | Object |
|---|---|---|
| `sample_01` | `camera` | Camera |
| `sample_02` | `headphone` | Headphones |
| `sample_03` | `battery` | Camera battery |
| `sample_04` | `chopstick` | Chopsticks |
| `sample_05` | `coin` | Coin |

All five rows are `test` inputs with capture group `unknown`; the pack does not assert an independent collection session.

## Contents

- `dataset.json`: dataset purpose, annotation status, training mode, and limitations.
- `samples.csv`: image metadata with `sample_id,image,label,split,group_id,sha256`.
- Image files referenced by the metadata: the five evaluation inputs.
- `training_features.csv`: fixed snapshot of the original `data/object_db.csv`.

The training snapshot has five reference samples, one each for `camera`, `headphone`, `battery`, `chopstick`, and `coin`. Its four values are the existing `shape4d` features: fill ratio, aspect ratio, and signed-log first and second Hu moments. These are compatible legacy fields, not features newly extracted by the refactored pipeline. The runner imports these vectors unchanged into a fresh database; it does not reapply their transform or train a neural network.

`training.kind: legacy_features` is accepted only with `purpose: smoke`. The snapshot has no traceable source-image or capture-session metadata, so independence between training and evaluation cannot be established. `owner_verified` applies to the image annotations and does not establish an independent split. SHA-256 checks preserve file identity, not statistical independence.

## Reproduce

From the repository root, build and test first:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 scripts/run_eval.py --out artifacts/evaluations/real-smoke-001
```

Read the generated `report.md` and `summary.json`. Use a new output directory for each run. You can validate paths, hashes, labels, and metadata without running the pipeline:

```sh
python3 scripts/run_eval.py --validate-only
```

A changed input requires an intentional dataset revision and updated metadata, not bypassing the checksum check. Do not tune parameters on these five images and then present their accuracy as unseen-data performance. Build a separately collected, capture-group-separated image dataset before making a held-out accuracy claim.

See [evaluation documentation](../../../docs/evaluation.md) for result statuses, timing definitions, and the future image-based train/test workflow.
