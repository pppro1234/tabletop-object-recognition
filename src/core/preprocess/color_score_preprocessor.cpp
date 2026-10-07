#include "factories.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <memory>

namespace vision {
namespace {

int clamp8(int value) {
    return std::min(255, std::max(0, value));
}

int makeOdd(int value) {
    int kernel = std::max(1, value);
    if (kernel % 2 == 0) {
        kernel += 1;
    }
    return kernel;
}

class ColorScorePreprocessor final : public Preprocessor {
public:
    explicit ColorScorePreprocessor(PreprocessConfig config) : config_(config) {}

    cv::Mat run(const cv::Mat &frame) const override {
        if (frame.empty() || frame.type() != CV_8UC3) {
            return {};
        }

        cv::Mat blurred;
        cv::GaussianBlur(frame, blurred, cv::Size(makeOdd(config_.blur_kernel_size), makeOdd(config_.blur_kernel_size)), 0.0);

        cv::Mat score(frame.size(), CV_8UC1);
        for (int y = 0; y < blurred.rows; ++y) {
            const cv::Vec3b *src = blurred.ptr<cv::Vec3b>(y);
            unsigned char *dst = score.ptr<unsigned char>(y);
            for (int x = 0; x < blurred.cols; ++x) {
                const int b = static_cast<int>(src[x][0]);
                const int g = static_cast<int>(src[x][1]);
                const int r = static_cast<int>(src[x][2]);
                const int intensity = (b + g + r) / 3;
                const int max_value = std::max({b, g, r});
                const int min_value = std::min({b, g, r});
                const int saturation = (max_value == 0) ? 0 : ((max_value - min_value) * 255) / max_value;
                const int score_value = intensity - ((std::max(0, config_.saturation_weight) * saturation) / 4);
                dst[x] = static_cast<unsigned char>(clamp8(score_value));
            }
        }
        return score;
    }

private:
    PreprocessConfig config_;
};

}  // namespace

std::unique_ptr<Preprocessor> makeColorScorePreprocessor(const PreprocessConfig &config) {
    return std::make_unique<ColorScorePreprocessor>(config);
}

}  // namespace vision
