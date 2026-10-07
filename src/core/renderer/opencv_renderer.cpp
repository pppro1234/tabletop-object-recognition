#include "factories.hpp"

#include <opencv2/imgproc.hpp>

#include <iomanip>
#include <memory>
#include <sstream>

namespace vision {
namespace {

cv::Scalar colorForId(int id) {
    cv::RNG rng(12345 + id * 7919);
    return cv::Scalar(rng.uniform(40, 255), rng.uniform(40, 255), rng.uniform(40, 255));
}

std::string formatScore(float score) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << score;
    return stream.str();
}

class OpenCvRenderer final : public ResultRenderer {
public:
    explicit OpenCvRenderer(RenderingConfig config) : config_(config) {}

    cv::Mat render(const FramePacket &frame, const PipelineResult &result) const override {
        if (frame.frame.empty()) {
            return {};
        }

        cv::Mat overlay = frame.frame.clone();
        for (const DetectionCandidate &candidate : result.candidates) {
            const Detection &detection = candidate.detection;
            const int color_id = (detection.track_id > 0) ? detection.track_id : detection.region_id;
            const cv::Scalar color = colorForId(color_id);

            cv::Point2f corners[4];
            detection.obb.points(corners);
            for (int i = 0; i < 4; ++i) {
                cv::line(overlay, corners[i], corners[(i + 1) % 4], color, 2, cv::LINE_AA);
            }

            cv::circle(overlay, detection.centroid, 3, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);

            if (config_.draw_labels) {
                const std::string text = "#" + std::to_string(detection.track_id) + " " + detection.label + " s=" +
                                         formatScore(detection.score) + " a=" + std::to_string(static_cast<int>(detection.angle_deg));
                cv::putText(overlay, text,
                            cv::Point(static_cast<int>(detection.centroid.x + 8), static_cast<int>(detection.centroid.y - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
            }
        }

        const std::string summary = "detections=" + std::to_string(result.candidates.size());
        cv::putText(overlay, summary, cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2,
                    cv::LINE_AA);
        return overlay;
    }

private:
    RenderingConfig config_;
};

}  // namespace

std::unique_ptr<ResultRenderer> makeOpenCvRenderer(const RenderingConfig &config) {
    return std::make_unique<OpenCvRenderer>(config);
}

}  // namespace vision
