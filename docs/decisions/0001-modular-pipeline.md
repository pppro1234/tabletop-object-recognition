# ADR 0001: Modular pipeline first

## Status

Accepted

## Context

An earlier standalone implementation concentrated video IO, thresholding, feature extraction, training, evaluation, and rendering inside one entrypoint. That made testing and reusing the recognition pipeline difficult.

## Decision

Adopt a modular pipeline with explicit interfaces for `FrameSource`, `Segmenter`, `RegionAnalyzer`, `FeatureExtractor`, `Classifier`, `Tracker`, `ResultRenderer`, and `RunStorage`.

## Consequences

- Easier unit and integration testing.
- Runtime behavior can be driven by config files.
- Serving and offline evaluation can reuse the same pipeline assembly.
