#pragma once

#include "vision/pipeline.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vision_test {

namespace fs = std::filesystem;

inline void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void requireNear(double actual, double expected, double tolerance, const std::string &message) {
    if (std::abs(actual - expected) > tolerance) {
        std::ostringstream stream;
        stream << message << " expected=" << expected << " actual=" << actual << " tolerance=" << tolerance;
        throw std::runtime_error(stream.str());
    }
}

inline double angleDistanceMod180(double lhs_deg, double rhs_deg) {
    double delta = std::fmod(std::abs(lhs_deg - rhs_deg), 180.0);
    if (delta < 0.0) {
        delta += 180.0;
    }
    return std::min(delta, 180.0 - delta);
}

inline fs::path sourceDir() {
#ifdef VISION_SOURCE_DIR
    return fs::path(VISION_SOURCE_DIR);
#else
    return fs::current_path();
#endif
}

inline fs::path tempRoot(const std::string &name) {
    const fs::path root = fs::temp_directory_path() / "tabletop_object_recognition_tests" / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    return root;
}

inline vision::AppConfig makeTestConfig(const std::string &name) {
    const fs::path root = tempRoot(name);

    vision::AppConfig config;
    config.run_name = name;
    config.input.mode = vision::InputMode::kBatchImages;
    config.input.image_glob = "*.ppm";
    config.preprocess.blur_kernel_size = 3;
    config.segmentation.sampling_stride = 1;
    config.segmentation.kmeans_iterations = 20;
    config.morphology.kernel_size = 3;
    config.morphology.open_iterations = 1;
    config.morphology.close_iterations = 1;
    config.region.min_area = 80;
    config.region.max_regions = 1;
    config.region.reject_boundary = true;
    config.classification.unknown_threshold = 100.0;
    config.storage.database_path = (root / "vision.sqlite3").string();
    config.storage.sample_image_dir = (root / "samples").string();
    config.evaluation.output_path = (root / "eval_results.csv").string();
    config.evaluation.manifest_path = (root / "manifest.csv").string();
    config.observability.artifact_root = (root / "artifacts").string();
    config.rendering.draw_debug_views = false;
    return config;
}

inline cv::Mat makeCanvas(int width = 160, int height = 120, unsigned char value = 255) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(value, value, value));
}

inline void drawFilledRotatedRect(cv::Mat &image, const cv::Point2f &center, const cv::Size2f &size, float angle_deg,
                                  const cv::Scalar &color = cv::Scalar(0, 0, 0)) {
    const cv::RotatedRect rect(center, size, angle_deg);
    cv::Point2f vertices[4];
    rect.points(vertices);

    std::vector<cv::Point> polygon;
    polygon.reserve(4);
    for (const cv::Point2f &vertex : vertices) {
        polygon.emplace_back(cv::Point(cvRound(vertex.x), cvRound(vertex.y)));
    }
    cv::fillConvexPoly(image, polygon, color, cv::LINE_AA);
}

inline vision::FramePacket makeFrame(const std::string &source_id, std::int64_t frame_index, const cv::Mat &image) {
    vision::FramePacket frame;
    frame.source_id = source_id;
    frame.frame_index = frame_index;
    frame.timestamp_ms = 0.0;
    frame.frame = image.clone();
    return frame;
}

inline std::vector<std::string> splitCsvLine(const std::string &line) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ',')) {
        parts.push_back(item);
    }
    return parts;
}

struct ExpectedDetection {
    std::string image;
    std::string label;
    double centroid_x = 0.0;
    double centroid_y = 0.0;
    double angle_deg = 0.0;
    cv::Rect bbox;
};

inline std::vector<ExpectedDetection> loadExpectedDetections(const fs::path &csv_path) {
    std::ifstream input(csv_path);
    require(input.is_open(), "Failed to open expected detection file: " + csv_path.string());

    std::vector<ExpectedDetection> expectations;
    std::string line;
    bool is_header = true;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        if (is_header) {
            is_header = false;
            continue;
        }

        const std::vector<std::string> parts = splitCsvLine(line);
        require(parts.size() >= 9, "Malformed expected detection row: " + line);

        ExpectedDetection expected;
        expected.image = parts[0];
        expected.label = parts[1];
        expected.centroid_x = std::stod(parts[2]);
        expected.centroid_y = std::stod(parts[3]);
        expected.angle_deg = std::stod(parts[4]);
        expected.bbox.x = std::stoi(parts[5]);
        expected.bbox.y = std::stoi(parts[6]);
        expected.bbox.width = std::stoi(parts[7]);
        expected.bbox.height = std::stoi(parts[8]);
        expectations.push_back(expected);
    }
    return expectations;
}

inline double intersectionOverUnion(const cv::Rect &lhs, const cv::Rect &rhs) {
    const cv::Rect overlap = lhs & rhs;
    const double intersection = static_cast<double>(overlap.area());
    const double union_area = static_cast<double>(lhs.area() + rhs.area() - overlap.area());
    if (union_area <= 0.0) {
        return 0.0;
    }
    return intersection / union_area;
}

inline double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = std::clamp(fraction, 0.0, 1.0) * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) {
        return values[lower];
    }
    const double alpha = position - static_cast<double>(lower);
    return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

inline double mean(const std::vector<double> &values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

}  // namespace vision_test
