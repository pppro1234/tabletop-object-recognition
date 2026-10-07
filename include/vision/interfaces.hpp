#pragma once

#include "vision/types.hpp"

#include <cstddef>
#include <vector>

namespace vision {

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual bool open(const InputConfig &config) = 0;
    virtual bool next(FramePacket &frame) = 0;
    virtual void close() = 0;
};

class Preprocessor {
public:
    virtual ~Preprocessor() = default;
    virtual cv::Mat run(const cv::Mat &frame) const = 0;
};

class Segmenter {
public:
    virtual ~Segmenter() = default;
    virtual cv::Mat segment(const cv::Mat &score_image) const = 0;
};

class MorphologyProcessor {
public:
    virtual ~MorphologyProcessor() = default;
    virtual cv::Mat refine(const cv::Mat &binary_mask) const = 0;
};

class RegionAnalyzer {
public:
    virtual ~RegionAnalyzer() = default;
    virtual std::vector<Region> analyze(const cv::Mat &binary_mask, cv::Mat *label_image) const = 0;
};

class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;
    virtual FeatureVector extract(const Region &region) const = 0;
};

class Classifier {
public:
    virtual ~Classifier() = default;
    virtual void reload() = 0;
    virtual ClassifierResult predict(const FeatureVector &feature) const = 0;
    virtual bool train(const TrainingSample &sample) = 0;
    virtual std::size_t sampleCount() const = 0;
};

class Tracker {
public:
    virtual ~Tracker() = default;
    virtual std::vector<Detection> assign(const std::vector<Detection> &detections) = 0;
    virtual void reset() = 0;
};

class ResultRenderer {
public:
    virtual ~ResultRenderer() = default;
    virtual cv::Mat render(const FramePacket &frame, const PipelineResult &result) const = 0;
};

class RunStorage {
public:
    virtual ~RunStorage() = default;
    virtual std::vector<TrainingSample> loadTrainingSamples() const = 0;
    virtual bool appendTrainingSample(const TrainingSample &sample) = 0;
    virtual std::int64_t beginRun(const RunRecord &run) = 0;
    virtual bool appendPredictions(std::int64_t run_id, const std::vector<EvaluationRecord> &records) = 0;
    virtual bool finishRun(std::int64_t run_id, double avg_fps, double accuracy) = 0;
    virtual bool exportConfusionMatrix(std::int64_t run_id, const std::string &csv_path) const = 0;
    virtual bool writeEvaluation(const std::vector<EvaluationRecord> &records) const = 0;
};

}  // namespace vision
