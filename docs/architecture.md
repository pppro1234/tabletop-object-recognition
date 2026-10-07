# Architecture

## Goals

- Organize recognition logic as a modular, maintainable software system.
- Decouple recognition logic from frame acquisition, persistence, rendering, and evaluation.
- Move thresholds and runtime behavior into configuration instead of hardcoded constants.

## Module responsibilities

- `FrameSource`: webcam or batch-image ingestion.
- `Preprocessor`: frame-to-score-image conversion.
- `Segmenter`: score-image to binary mask.
- `MorphologyProcessor`: cleanup and hole filling.
- `RegionAnalyzer`: connected components and geometric region properties.
- `FeatureExtractor`: invariant feature vector construction.
- `Classifier`: prediction and training-sample persistence.
- `Tracker`: stable track IDs across live frames.
- `ResultRenderer`: OpenCV overlays for operators and demos.
- `RunStorage`: SQLite-backed training/evaluation persistence for `object_samples`, `runs`, and `predictions`.
- `Pipeline`: orchestration only; no UI or file-format policy.

## Current runtime flow

`FrameSource -> Preprocessor -> Segmenter -> Morphology -> RegionAnalyzer -> FeatureExtractor -> Classifier -> Tracker -> ResultRenderer`

## Offline evaluation

`tabletop_recognition_eval` consumes an explicit labeled manifest and records one outcome per entry, including missing
or unreadable images. It does not use directory scanning or filename-derived labels. The Python runner
freezes the inputs and configuration, isolates the training database, validates capture groups and content
hashes, and records per-image status and latency. See [evaluation.md](evaluation.md) for the data contract.

Synthetic fixtures validate software regressions. The separate five-image real dataset validates a fixed
smoke scenario; its historical training vectors do not establish independent holdout performance.

ResNet extraction, richer tracking, and the HTTP prediction endpoint remain outside the current pipeline.
