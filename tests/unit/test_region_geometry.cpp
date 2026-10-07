#include "factories.hpp"
#include "test_support.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    vision::RegionConfig config;
    config.min_area = 100;
    config.max_regions = 1;
    config.reject_boundary = true;

    std::unique_ptr<vision::RegionAnalyzer> analyzer = vision::makeConnectedComponentAnalyzer(config);

    cv::Mat mask = cv::Mat::zeros(160, 160, CV_8UC1);
    const cv::RotatedRect target(cv::Point2f(82.0f, 78.0f), cv::Size2f(84.0f, 26.0f), 33.0f);
    cv::Point2f vertices[4];
    target.points(vertices);

    std::vector<cv::Point> polygon;
    polygon.reserve(4);
    for (const cv::Point2f &vertex : vertices) {
        polygon.emplace_back(cv::Point(cvRound(vertex.x), cvRound(vertex.y)));
    }
    cv::fillConvexPoly(mask, polygon, cv::Scalar(255), cv::LINE_AA);

    cv::Mat labels_first;
    cv::Mat labels_second;
    const std::vector<vision::Region> first = analyzer->analyze(mask, &labels_first);
    const std::vector<vision::Region> second = analyzer->analyze(mask, &labels_second);

    vision_test::require(first.size() == 1, "Analyzer should find one elongated component.");
    vision_test::require(second.size() == 1, "Analyzer should be deterministic across repeated runs.");

    const vision::Region &region = first.front();
    vision_test::requireNear(region.centroid.x, 82.0, 2.0, "Centroid x should stay stable.");
    vision_test::requireNear(region.centroid.y, 78.0, 2.0, "Centroid y should stay stable.");
    vision_test::require(vision_test::angleDistanceMod180(region.angle_deg, 33.0) <= 6.0,
                         "Moment-based angle should track the major axis.");
    vision_test::require(vision_test::angleDistanceMod180(region.angle_deg, second.front().angle_deg) <= 0.1,
                         "Angle should be stable across repeated analyses.");
    return 0;
}
