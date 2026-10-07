#include "factories.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace vision {
namespace {

std::vector<int> selectMajorRegions(const cv::Mat &stats, cv::Size image_size, const RegionConfig &config) {
    std::vector<std::pair<int, int>> candidates;
    for (int label = 1; label < stats.rows; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < std::max(1, config.min_area)) {
            continue;
        }

        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const bool touches_boundary =
            left <= 0 || top <= 0 || left + width >= image_size.width || top + height >= image_size.height;
        if (config.reject_boundary && touches_boundary) {
            continue;
        }
        candidates.emplace_back(label, area);
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) { return lhs.second > rhs.second; });
    const int keep_count = std::max(1, config.max_regions);
    if (static_cast<int>(candidates.size()) > keep_count) {
        candidates.resize(static_cast<std::size_t>(keep_count));
    }

    std::vector<int> labels;
    labels.reserve(candidates.size());
    for (const auto &[label, _] : candidates) {
        labels.push_back(label);
    }
    return labels;
}

Region computeRegion(const cv::Mat &labels, const cv::Mat &stats, int label) {
    Region region;
    region.region_id = label;

    cv::Mat mask;
    cv::compare(labels, label, mask, cv::CMP_EQ);

    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    if (points.empty()) {
        return region;
    }

    const cv::Moments moments = cv::moments(mask, true);
    if (moments.m00 <= std::numeric_limits<double>::epsilon()) {
        return region;
    }

    region.centroid = cv::Point2f(static_cast<float>(moments.m10 / moments.m00),
                                  static_cast<float>(moments.m01 / moments.m00));
    const double angle_rad = 0.5 * std::atan2(2.0 * moments.mu11, moments.mu20 - moments.mu02);
    region.angle_deg = static_cast<float>(angle_rad * 180.0 / CV_PI);
    region.obb = cv::minAreaRect(points);

    const cv::Point2f e1(static_cast<float>(std::cos(angle_rad)), static_cast<float>(std::sin(angle_rad)));
    const cv::Point2f e2(static_cast<float>(std::sin(angle_rad)), static_cast<float>(-std::cos(angle_rad)));

    double min_e1 = std::numeric_limits<double>::max();
    double max_e1 = -std::numeric_limits<double>::max();
    double min_e2 = std::numeric_limits<double>::max();
    double max_e2 = -std::numeric_limits<double>::max();
    for (const cv::Point &point : points) {
        const cv::Point2f delta(static_cast<float>(point.x) - region.centroid.x,
                                static_cast<float>(point.y) - region.centroid.y);
        const double proj1 = delta.x * e1.x + delta.y * e1.y;
        const double proj2 = delta.x * e2.x + delta.y * e2.y;
        min_e1 = std::min(min_e1, proj1);
        max_e1 = std::max(max_e1, proj1);
        min_e2 = std::min(min_e2, proj2);
        max_e2 = std::max(max_e2, proj2);
    }

    region.min_e1 = static_cast<float>(min_e1);
    region.max_e1 = static_cast<float>(max_e1);
    region.min_e2 = static_cast<float>(min_e2);
    region.max_e2 = static_cast<float>(max_e2);

    region.area = stats.at<int>(label, cv::CC_STAT_AREA);
    region.bounding_box = cv::Rect(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                                   stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
    region.touches_boundary = region.bounding_box.x <= 0 || region.bounding_box.y <= 0 ||
                              region.bounding_box.x + region.bounding_box.width >= labels.cols ||
                              region.bounding_box.y + region.bounding_box.height >= labels.rows;

    const double box_width = std::max(1.0f, region.obb.size.width);
    const double box_height = std::max(1.0f, region.obb.size.height);
    region.fill_ratio = static_cast<double>(region.area) / (box_width * box_height);
    region.aspect_ratio = std::max(box_width, box_height) / std::max(1.0, std::min(box_width, box_height));

    double hu[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    cv::HuMoments(moments, hu);
    region.hu1 = hu[0];
    region.hu2 = hu[1];
    return region;
}

class ConnectedComponentAnalyzer final : public RegionAnalyzer {
public:
    explicit ConnectedComponentAnalyzer(RegionConfig config) : config_(config) {}

    std::vector<Region> analyze(const cv::Mat &binary_mask, cv::Mat *label_image) const override {
        if (binary_mask.empty() || binary_mask.type() != CV_8UC1) {
            return {};
        }

        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int count = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);
        if (count <= 1) {
            if (label_image != nullptr) {
                *label_image = labels;
            }
            return {};
        }

        if (label_image != nullptr) {
            *label_image = labels;
        }

        std::vector<Region> regions;
        for (int label : selectMajorRegions(stats, binary_mask.size(), config_)) {
            Region region = computeRegion(labels, stats, label);
            if (region.region_id > 0) {
                regions.push_back(region);
            }
        }
        return regions;
    }

private:
    RegionConfig config_;
};

}  // namespace

std::unique_ptr<RegionAnalyzer> makeConnectedComponentAnalyzer(const RegionConfig &config) {
    return std::make_unique<ConnectedComponentAnalyzer>(config);
}

}  // namespace vision
