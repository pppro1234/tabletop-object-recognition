#pragma once

#include "vision/interfaces.hpp"

#include <memory>
#include <string>

namespace vision {

class Pipeline {
public:
    Pipeline(AppConfig config,
             std::unique_ptr<Preprocessor> preprocessor,
             std::unique_ptr<Segmenter> segmenter,
             std::unique_ptr<MorphologyProcessor> morphology,
             std::unique_ptr<RegionAnalyzer> region_analyzer,
             std::unique_ptr<FeatureExtractor> feature_extractor,
             std::unique_ptr<Classifier> classifier,
             std::unique_ptr<Tracker> tracker,
             std::unique_ptr<ResultRenderer> renderer);

    PipelineResult run(const FramePacket &frame);
    bool storeTrainingSample(const FramePacket &frame, const PipelineResult &result, std::size_t candidate_index,
                             const std::string &label, std::string *error = nullptr);

    const AppConfig &config() const;
    Classifier &classifier();
    const Classifier &classifier() const;

private:
    AppConfig config_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<Segmenter> segmenter_;
    std::unique_ptr<MorphologyProcessor> morphology_;
    std::unique_ptr<RegionAnalyzer> region_analyzer_;
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    std::unique_ptr<Classifier> classifier_;
    std::unique_ptr<Tracker> tracker_;
    std::unique_ptr<ResultRenderer> renderer_;
};

AppConfig loadConfig(const std::string &path);
std::unique_ptr<FrameSource> makeFrameSource(const AppConfig &config);
std::unique_ptr<RunStorage> makeRunStorage(const AppConfig &config);
std::unique_ptr<Pipeline> makeDefaultPipeline(const AppConfig &config, RunStorage &storage);

}  // namespace vision
