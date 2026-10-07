#include "factories.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <memory>

namespace vision {
namespace {

int makeOdd(int value) {
    int kernel = std::max(1, value);
    if (kernel % 2 == 0) {
        kernel += 1;
    }
    return kernel;
}

class BasicMorphologyProcessor final : public MorphologyProcessor {
public:
    explicit BasicMorphologyProcessor(MorphologyConfig config) : config_(config) {}

    cv::Mat refine(const cv::Mat &binary_mask) const override {
        if (binary_mask.empty() || binary_mask.type() != CV_8UC1) {
            return {};
        }

        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(makeOdd(config_.kernel_size), makeOdd(config_.kernel_size)));

        cv::Mat opened;
        cv::Mat cleaned;
        cv::morphologyEx(binary_mask, opened, cv::MORPH_OPEN, kernel, cv::Point(-1, -1),
                         std::max(0, config_.open_iterations));
        cv::morphologyEx(opened, cleaned, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1),
                         std::max(0, config_.close_iterations));
        return cleaned;
    }

private:
    MorphologyConfig config_;
};

}  // namespace

std::unique_ptr<MorphologyProcessor> makeBasicMorphologyProcessor(const MorphologyConfig &config) {
    return std::make_unique<BasicMorphologyProcessor>(config);
}

}  // namespace vision
