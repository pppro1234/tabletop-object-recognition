#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace vision {

enum class InputMode {
    kWebcam,
    kBatchImages,
};

struct FramePacket {
    std::string source_id;
    std::int64_t frame_index = 0;
    double timestamp_ms = 0.0;
    cv::Mat frame;
};

struct Region {
    int region_id = -1;
    int area = 0;
    cv::Rect bounding_box;
    cv::Point2f centroid{0.0f, 0.0f};
    cv::RotatedRect obb;
    float angle_deg = 0.0f;
    bool touches_boundary = false;
    double fill_ratio = 0.0;
    double aspect_ratio = 0.0;
    double hu1 = 0.0;
    double hu2 = 0.0;
    float min_e1 = 0.0f;
    float max_e1 = 0.0f;
    float min_e2 = 0.0f;
    float max_e2 = 0.0f;
};

struct FeatureVector {
    std::string extractor = "shape4d";
    std::vector<double> values;
};

struct ClassifierResult {
    std::string label = "Unknown";
    float score = 0.0f;
    double distance = std::numeric_limits<double>::infinity();
};

struct Detection {
    int track_id = -1;
    std::string label;
    float score = 0.0f;
    cv::Point2f centroid;
    float angle_deg = 0.0f;
    cv::RotatedRect obb;
    double latency_ms = 0.0;
    int region_id = -1;
    cv::Rect bounding_box;
};

struct TrainingSample {
    std::string sample_id;
    std::string label;
    FeatureVector feature;
    std::vector<double> embedding;
    std::string image_path;
    std::string created_at;
    cv::Point2f centroid;
    float angle_deg = 0.0f;
    std::string source_id;
};

struct EvaluationRecord {
    std::string frame_id;
    std::string true_label;
    std::string predicted_label;
    float score = 0.0f;
    double latency_ms = 0.0;
    bool correct = false;
    std::string status = "ok";
    int detection_count = 0;
};

struct RunRecord {
    std::int64_t id = 0;
    std::string config_name;
    std::string git_commit;
    std::string dataset_name;
    std::string started_at;
    double avg_fps = 0.0;
    double accuracy = 0.0;
};

struct DetectionCandidate {
    Region region;
    FeatureVector feature;
    Detection detection;
};

struct PipelineArtifacts {
    cv::Mat score_image;
    cv::Mat binary_mask;
    cv::Mat cleaned_mask;
    cv::Mat region_map;
    cv::Mat overlay;
};

struct PipelineTimings {
    double preprocess_ms = 0.0;
    double segmentation_ms = 0.0;
    double morphology_ms = 0.0;
    double components_ms = 0.0;
    double feature_ms = 0.0;
    double classification_ms = 0.0;
    double tracking_ms = 0.0;
    double rendering_ms = 0.0;
    double total_ms = 0.0;
};

struct PipelineResult {
    std::vector<DetectionCandidate> candidates;
    PipelineArtifacts artifacts;
    PipelineTimings timings;
};

struct PreprocessConfig {
    int blur_kernel_size = 5;
    int saturation_weight = 2;
};

struct SegmentationConfig {
    int threshold_offset = 0;
    int kmeans_iterations = 15;
    int sampling_stride = 4;
};

struct MorphologyConfig {
    int kernel_size = 5;
    int open_iterations = 1;
    int close_iterations = 1;
};

struct RegionConfig {
    int min_area = 500;
    int max_regions = 5;
    bool reject_boundary = true;
};

struct ClassificationConfig {
    double unknown_threshold = 4.0;
};

struct StorageConfig {
    std::string database_path = "data/db/vision.sqlite3";
    std::string sample_image_dir = "data/samples/captured";
};

struct TrackingConfig {
    float max_match_distance = 80.0f;
};

struct RenderingConfig {
    bool draw_debug_views = true;
    bool draw_labels = true;
};

struct InputConfig {
    InputMode mode = InputMode::kWebcam;
    int webcam_index = 0;
    std::string image_dir = "data/samples";
    std::string image_glob = "*.jpg";
};

struct EvaluationConfig {
    std::string manifest_path = "data/db/eval_manifest.csv";
    std::string output_path = "data/db/eval_results.csv";
};

struct ObservabilityConfig {
    std::string artifact_root = "artifacts";
    bool save_failure_samples = true;
    bool save_config_snapshot = true;
    bool log_per_frame = true;
};

struct AppConfig {
    std::string run_name = "default";
    std::vector<std::string> labels;
    InputConfig input;
    PreprocessConfig preprocess;
    SegmentationConfig segmentation;
    MorphologyConfig morphology;
    RegionConfig region;
    ClassificationConfig classification;
    StorageConfig storage;
    TrackingConfig tracking;
    RenderingConfig rendering;
    EvaluationConfig evaluation;
    ObservabilityConfig observability;
};

}  // namespace vision
