#pragma once

#include "vision/pipeline.hpp"

#include <memory>

namespace vision {

std::unique_ptr<FrameSource> makeWebcamFrameSource();
std::unique_ptr<FrameSource> makeBatchImageFrameSource();
std::unique_ptr<Preprocessor> makeColorScorePreprocessor(const PreprocessConfig &config);
std::unique_ptr<Segmenter> makeKMeansSegmenter(const SegmentationConfig &config);
std::unique_ptr<MorphologyProcessor> makeBasicMorphologyProcessor(const MorphologyConfig &config);
std::unique_ptr<RegionAnalyzer> makeConnectedComponentAnalyzer(const RegionConfig &config);
std::unique_ptr<FeatureExtractor> makeShapeFeatureExtractor();
std::unique_ptr<Classifier> makeNearestNeighborClassifier(const ClassificationConfig &config, RunStorage &storage);
std::unique_ptr<Tracker> makeCentroidTracker(const TrackingConfig &config);
std::unique_ptr<ResultRenderer> makeOpenCvRenderer(const RenderingConfig &config);
std::unique_ptr<RunStorage> makeSqliteRunStorage(const AppConfig &config);

}  // namespace vision
