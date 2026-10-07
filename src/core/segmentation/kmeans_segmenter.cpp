#include "factories.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

namespace vision {
namespace {

int clamp8(int value) {
    return std::min(255, std::max(0, value));
}

class KMeansSegmenter final : public Segmenter {
public:
    explicit KMeansSegmenter(SegmentationConfig config) : config_(config) {}

    cv::Mat segment(const cv::Mat &score_image) const override {
        if (score_image.empty() || score_image.type() != CV_8UC1) {
            return {};
        }

        const int stride = std::max(1, config_.sampling_stride);
        std::vector<int> samples;
        samples.reserve((score_image.rows * score_image.cols) / (stride * stride) + 1);

        int min_value = 255;
        int max_value = 0;
        for (int y = 0; y < score_image.rows; y += stride) {
            const unsigned char *row = score_image.ptr<unsigned char>(y);
            for (int x = 0; x < score_image.cols; x += stride) {
                const int value = static_cast<int>(row[x]);
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
                samples.push_back(value);
            }
        }

        int threshold = 128;
        if (!samples.empty()) {
            if (min_value == max_value) {
                threshold = min_value;
            } else {
                float m1 = static_cast<float>(min_value);
                float m2 = static_cast<float>(max_value);
                for (int iter = 0; iter < std::max(1, config_.kmeans_iterations); ++iter) {
                    long long sum1 = 0;
                    long long sum2 = 0;
                    int count1 = 0;
                    int count2 = 0;
                    for (int sample : samples) {
                        if (std::abs(sample - m1) <= std::abs(sample - m2)) {
                            sum1 += sample;
                            ++count1;
                        } else {
                            sum2 += sample;
                            ++count2;
                        }
                    }

                    if (count1 == 0 || count2 == 0) {
                        threshold = static_cast<int>(std::accumulate(samples.begin(), samples.end(), 0LL) / samples.size());
                        break;
                    }

                    const float next_m1 = static_cast<float>(sum1) / static_cast<float>(count1);
                    const float next_m2 = static_cast<float>(sum2) / static_cast<float>(count2);
                    if (std::abs(next_m1 - m1) < 0.5f && std::abs(next_m2 - m2) < 0.5f) {
                        m1 = next_m1;
                        m2 = next_m2;
                        break;
                    }
                    m1 = next_m1;
                    m2 = next_m2;
                }
                if (m1 > m2) {
                    std::swap(m1, m2);
                }
                threshold = static_cast<int>(std::lround((m1 + m2) * 0.5f));
            }
        }

        threshold = clamp8(threshold + config_.threshold_offset);
        cv::Mat binary(score_image.size(), CV_8UC1);
        for (int y = 0; y < score_image.rows; ++y) {
            const unsigned char *src = score_image.ptr<unsigned char>(y);
            unsigned char *dst = binary.ptr<unsigned char>(y);
            for (int x = 0; x < score_image.cols; ++x) {
                dst[x] = (src[x] <= threshold) ? 255 : 0;
            }
        }
        return binary;
    }

private:
    SegmentationConfig config_;
};

}  // namespace

std::unique_ptr<Segmenter> makeKMeansSegmenter(const SegmentationConfig &config) {
    return std::make_unique<KMeansSegmenter>(config);
}

}  // namespace vision
