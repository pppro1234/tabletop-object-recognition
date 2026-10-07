The SQLite experiment database, evaluation manifests, and exported metrics live here.

Default database:

```text
data/db/vision.sqlite3
```

Core tables:
- `object_samples`: training samples with serialized feature vectors, optional embeddings, image crop paths, and timestamps
- `runs`: one row per evaluation run with config name, git commit, dataset name, start time, average FPS, and accuracy
- `predictions`: one row per evaluated frame with true label, predicted label, score, latency, status, and detection count

Opening an old database adds `status` and `detection_count` without replacing existing rows. Historical rows
receive status `legacy` because their detection/error state cannot be reconstructed.

The historical `runs.accuracy` column and confusion CSV's `accuracy_percent` use percentages (0–100).
JSON `accuracy` values use fractions (0–1). New code retains these units to avoid reinterpreting old rows.

For reproducible evaluation, use `scripts/run_eval.py`. It creates a fresh database inside each experiment
directory instead of reusing this local working database, whose old `demo`/`sample` rows are not a benchmark.

The evaluation binary also exports file-based artifacts beside the database:
- `eval_results.csv`
- `metrics.json`
- `confusion_matrix.csv`
