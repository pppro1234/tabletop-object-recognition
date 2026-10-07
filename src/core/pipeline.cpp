#include "vision/pipeline.hpp"

#include "factories.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vision {
namespace {

namespace fs = std::filesystem;

std::string readString(const cv::FileNode &node, const char *key, const std::string &fallback) {
    if (node.empty() || node[key].empty()) {
        return fallback;
    }
    return static_cast<std::string>(node[key]);
}

int readInt(const cv::FileNode &node, const char *key, int fallback) {
    if (node.empty() || node[key].empty()) {
        return fallback;
    }
    return static_cast<int>(node[key]);
}

double readDouble(const cv::FileNode &node, const char *key, double fallback) {
    if (node.empty() || node[key].empty()) {
        return fallback;
    }
    return static_cast<double>(node[key]);
}

bool readBool(const cv::FileNode &node, const char *key, bool fallback) {
    if (node.empty() || node[key].empty()) {
        return fallback;
    }
    return static_cast<int>(node[key]) != 0;
}

InputMode parseInputMode(const std::string &value) {
    return (value == "batch_images") ? InputMode::kBatchImages : InputMode::kWebcam;
}

std::string resolvePathString(const fs::path &config_dir, const std::string &raw_path) {
    if (raw_path.empty()) {
        return raw_path;
    }
    const fs::path path(raw_path);
    if (path.is_absolute()) {
        return path.string();
    }
    return (config_dir / path).lexically_normal().string();
}

std::string isoTimestampUtc() {
    const std::time_t raw_time = std::time(nullptr);
    std::tm time_info{};
#if defined(_WIN32)
    gmtime_s(&time_info, &raw_time);
#else
    gmtime_r(&raw_time, &time_info);
#endif

    std::ostringstream stream;
    stream << std::put_time(&time_info, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::string sanitizeToken(std::string value) {
    for (char &ch : value) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    if (value.empty()) {
        return "item";
    }
    return value;
}

std::string sampleSourceToken(const FramePacket &frame) {
    std::string token = fs::path(frame.source_id).filename().string();
    if (token.empty()) {
        token = frame.source_id;
    }
    return sanitizeToken(token);
}

bool saveTrainingImage(const AppConfig &config, const FramePacket &frame, const DetectionCandidate &candidate,
                       const std::string &label, const std::string &sample_id, std::string &image_path,
                       fs::path &created_dir, std::string &error) {
    if (frame.frame.empty()) {
        error = "input frame is empty";
        return false;
    }

    std::error_code ec;
    const fs::path label_dir = fs::path(config.storage.sample_image_dir) / sanitizeToken(label);
    fs::create_directories(label_dir, ec);
    if (ec) {
        error = "cannot create sample directory " + label_dir.string() + ": " + ec.message();
        return false;
    }

    const cv::Rect bounds(0, 0, frame.frame.cols, frame.frame.rows);
    const cv::Rect roi = candidate.detection.bounding_box & bounds;
    const cv::Mat crop = (roi.area() > 0) ? frame.frame(roi).clone() : frame.frame.clone();

    // Creating a directory is exclusive, so even concurrent saves cannot claim an existing sample's path.
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 100; ++attempt) {
        const fs::path path = label_dir / (sanitizeToken(sample_id) + "_" + std::to_string(nonce) + "_" +
                                           std::to_string(attempt));
        ec.clear();
        if (fs::create_directory(path, ec)) {
            created_dir = path;
            break;
        }
        if (ec && ec != std::errc::file_exists) {
            error = "cannot reserve sample path " + path.string() + ": " + ec.message();
            return false;
        }
    }
    if (created_dir.empty()) {
        error = "could not reserve a unique sample path in " + label_dir.string();
        return false;
    }

    const fs::path output_path = created_dir / "crop.png";
    try {
        if (!cv::imwrite(output_path.string(), crop)) {
            error = "OpenCV could not write " + output_path.string();
        }
    } catch (const std::exception &exception) {
        error = "OpenCV could not write " + output_path.string() + ": " + exception.what();
    }
    if (!error.empty()) {
        fs::remove_all(created_dir, ec);
        if (ec) {
            error += "; cleanup failed: " + ec.message();
        }
        created_dir.clear();
        return false;
    }
    image_path = output_path.string();
    return true;
}

cv::Scalar colorFromRegion(int region_id) {
    cv::RNG rng(98765 + region_id * 1237);
    return cv::Scalar(rng.uniform(40, 255), rng.uniform(40, 255), rng.uniform(40, 255));
}

cv::Mat buildRegionMap(const cv::Mat &labels, const std::vector<DetectionCandidate> &candidates) {
    if (labels.empty() || labels.type() != CV_32S) {
        return {};
    }

    cv::Mat region_map = cv::Mat::zeros(labels.size(), CV_8UC3);
    std::vector<int> lookup;
    int max_label = 0;
    for (const DetectionCandidate &candidate : candidates) {
        max_label = std::max(max_label, candidate.region.region_id);
    }
    lookup.assign(static_cast<std::size_t>(max_label + 1), -1);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const int label = candidates[i].region.region_id;
        if (label >= 0) {
            lookup[static_cast<std::size_t>(label)] = static_cast<int>(i);
        }
    }

    for (int y = 0; y < labels.rows; ++y) {
        const int *src = labels.ptr<int>(y);
        cv::Vec3b *dst = region_map.ptr<cv::Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) {
            const int label = src[x];
            if (label < 0 || label > max_label) {
                continue;
            }
            const int index = lookup[static_cast<std::size_t>(label)];
            if (index < 0) {
                continue;
            }
            const cv::Scalar color = colorFromRegion(candidates[static_cast<std::size_t>(index)].region.region_id);
            dst[x] = cv::Vec3b(static_cast<unsigned char>(color[0]), static_cast<unsigned char>(color[1]),
                               static_cast<unsigned char>(color[2]));
        }
    }
    return region_map;
}

}  // namespace

Pipeline::Pipeline(AppConfig config, std::unique_ptr<Preprocessor> preprocessor, std::unique_ptr<Segmenter> segmenter,
                   std::unique_ptr<MorphologyProcessor> morphology, std::unique_ptr<RegionAnalyzer> region_analyzer,
                   std::unique_ptr<FeatureExtractor> feature_extractor, std::unique_ptr<Classifier> classifier,
                   std::unique_ptr<Tracker> tracker, std::unique_ptr<ResultRenderer> renderer)
    : config_(std::move(config)),
      preprocessor_(std::move(preprocessor)),
      segmenter_(std::move(segmenter)),
      morphology_(std::move(morphology)),
      region_analyzer_(std::move(region_analyzer)),
      feature_extractor_(std::move(feature_extractor)),
      classifier_(std::move(classifier)),
      tracker_(std::move(tracker)),
      renderer_(std::move(renderer)) {}

PipelineResult Pipeline::run(const FramePacket &frame) {
    PipelineResult result;
    if (frame.frame.empty()) {
        return result;
    }

    using clock = std::chrono::steady_clock;
    const auto measureMs = [](clock::time_point begin, clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    const auto start = clock::now();
    auto stage_begin = start;
    result.artifacts.score_image = preprocessor_->run(frame.frame);
    auto stage_end = clock::now();
    result.timings.preprocess_ms = measureMs(stage_begin, stage_end);

    stage_begin = stage_end;
    result.artifacts.binary_mask = segmenter_->segment(result.artifacts.score_image);
    stage_end = clock::now();
    result.timings.segmentation_ms = measureMs(stage_begin, stage_end);

    stage_begin = stage_end;
    result.artifacts.cleaned_mask = morphology_->refine(result.artifacts.binary_mask);
    stage_end = clock::now();
    result.timings.morphology_ms = measureMs(stage_begin, stage_end);

    cv::Mat label_image;
    stage_begin = stage_end;
    const std::vector<Region> regions = region_analyzer_->analyze(result.artifacts.cleaned_mask, &label_image);
    stage_end = clock::now();
    result.timings.components_ms = measureMs(stage_begin, stage_end);

    std::vector<Detection> detections;
    detections.reserve(regions.size());

    for (const Region &region : regions) {
        DetectionCandidate candidate;
        candidate.region = region;

        stage_begin = clock::now();
        candidate.feature = feature_extractor_->extract(region);
        stage_end = clock::now();
        result.timings.feature_ms += measureMs(stage_begin, stage_end);

        stage_begin = clock::now();
        const ClassifierResult prediction = classifier_->predict(candidate.feature);
        stage_end = clock::now();
        result.timings.classification_ms += measureMs(stage_begin, stage_end);

        candidate.detection.label = prediction.label;
        candidate.detection.score = prediction.score;
        candidate.detection.centroid = region.centroid;
        candidate.detection.angle_deg = region.angle_deg;
        candidate.detection.obb = region.obb;
        candidate.detection.region_id = region.region_id;
        candidate.detection.bounding_box = region.bounding_box;

        detections.push_back(candidate.detection);
        result.candidates.push_back(std::move(candidate));
    }

    stage_begin = clock::now();
    const std::vector<Detection> tracked = tracker_->assign(detections);
    stage_end = clock::now();
    result.timings.tracking_ms = measureMs(stage_begin, stage_end);

    for (std::size_t i = 0; i < result.candidates.size() && i < tracked.size(); ++i) {
        result.candidates[i].detection = tracked[i];
    }

    stage_begin = clock::now();
    result.artifacts.region_map = buildRegionMap(label_image, result.candidates);
    result.artifacts.overlay = renderer_->render(frame, result);
    stage_end = clock::now();
    result.timings.rendering_ms = measureMs(stage_begin, stage_end);

    const double latency_ms = measureMs(start, stage_end);
    result.timings.total_ms = latency_ms;
    for (DetectionCandidate &candidate : result.candidates) {
        candidate.detection.latency_ms = latency_ms;
    }

    return result;
}

bool Pipeline::storeTrainingSample(const FramePacket &frame, const PipelineResult &result, std::size_t candidate_index,
                                   const std::string &label, std::string *error) {
    if (error != nullptr) {
        error->clear();
    }
    const auto fail = [error](const std::string &message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (label.empty()) {
        return fail("training label is empty");
    }
    if (candidate_index >= result.candidates.size()) {
        return fail("training candidate index is out of range");
    }
    if (result.candidates[candidate_index].feature.values.empty()) {
        return fail("training feature vector is empty");
    }
    if (frame.frame.empty()) {
        return fail("training input frame is empty");
    }

    const DetectionCandidate &candidate = result.candidates[candidate_index];
    const std::string sample_id =
        sanitizeToken(sampleSourceToken(frame)) + "_" + std::to_string(frame.frame_index) + "_" + std::to_string(candidate.region.region_id);

    TrainingSample sample;
    sample.label = label;
    sample.feature = candidate.feature;
    sample.embedding = {};
    sample.centroid = candidate.region.centroid;
    sample.angle_deg = candidate.region.angle_deg;
    sample.source_id = frame.source_id;
    sample.created_at = isoTimestampUtc();

    fs::path created_dir;
    std::string save_error;
    if (!saveTrainingImage(config_, frame, candidate, label, sample_id, sample.image_path, created_dir, save_error)) {
        return fail("Failed to save training image: " + save_error);
    }
    sample.sample_id = created_dir.filename().string();

    try {
        if (classifier_->train(sample)) {
            return true;
        }
        save_error = "Database rejected the training sample";
    } catch (const std::exception &exception) {
        save_error = std::string("Failed to write training sample to database: ") + exception.what();
    }

    // Only remove the directory reserved for this call; an older sample can never be its cleanup target.
    std::error_code ec;
    fs::remove_all(created_dir, ec);
    if (ec) {
        save_error += "; cleanup failed for " + created_dir.string() + ": " + ec.message();
    }
    return fail(save_error);
}

const AppConfig &Pipeline::config() const {
    return config_;
}

Classifier &Pipeline::classifier() {
    return *classifier_;
}

const Classifier &Pipeline::classifier() const {
    return *classifier_;
}

AppConfig loadConfig(const std::string &path) {
    cv::FileStorage storage(path, cv::FileStorage::READ);
    if (!storage.isOpened()) {
        throw std::runtime_error("Failed to open config: " + path);
    }

    const fs::path config_dir = fs::path(path).parent_path();
    AppConfig config;
    config.run_name = readString(storage.root(), "run_name", config.run_name);

    const cv::FileNode labels_node = storage["labels"];
    if (!labels_node.empty()) {
        for (const cv::FileNode &label : labels_node) {
            config.labels.push_back(static_cast<std::string>(label));
        }
    }

    const cv::FileNode input = storage["input"];
    config.input.mode = parseInputMode(readString(input, "mode", "webcam"));
    config.input.webcam_index = readInt(input, "webcam_index", config.input.webcam_index);
    config.input.image_dir = resolvePathString(config_dir, readString(input, "image_dir", config.input.image_dir));
    config.input.image_glob = readString(input, "image_glob", config.input.image_glob);

    const cv::FileNode preprocess = storage["preprocess"];
    config.preprocess.blur_kernel_size = readInt(preprocess, "blur_kernel_size", config.preprocess.blur_kernel_size);
    config.preprocess.saturation_weight =
        readInt(preprocess, "saturation_weight", config.preprocess.saturation_weight);

    const cv::FileNode segmentation = storage["segmentation"];
    config.segmentation.threshold_offset =
        readInt(segmentation, "threshold_offset", config.segmentation.threshold_offset);
    config.segmentation.kmeans_iterations =
        readInt(segmentation, "kmeans_iterations", config.segmentation.kmeans_iterations);
    config.segmentation.sampling_stride =
        readInt(segmentation, "sampling_stride", config.segmentation.sampling_stride);

    const cv::FileNode morphology = storage["morphology"];
    config.morphology.kernel_size = readInt(morphology, "kernel_size", config.morphology.kernel_size);
    config.morphology.open_iterations = readInt(morphology, "open_iterations", config.morphology.open_iterations);
    config.morphology.close_iterations = readInt(morphology, "close_iterations", config.morphology.close_iterations);

    const cv::FileNode region = storage["region"];
    config.region.min_area = readInt(region, "min_area", config.region.min_area);
    config.region.max_regions = readInt(region, "max_regions", config.region.max_regions);
    config.region.reject_boundary = readBool(region, "reject_boundary", config.region.reject_boundary);

    const cv::FileNode classification = storage["classification"];
    config.classification.unknown_threshold =
        readDouble(classification, "unknown_threshold", config.classification.unknown_threshold);

    const cv::FileNode storage_node = storage["storage"];
    config.storage.database_path =
        resolvePathString(config_dir, readString(storage_node, "database_path", config.storage.database_path));
    config.storage.sample_image_dir =
        resolvePathString(config_dir, readString(storage_node, "sample_image_dir", config.storage.sample_image_dir));

    const cv::FileNode tracking = storage["tracking"];
    config.tracking.max_match_distance =
        static_cast<float>(readDouble(tracking, "max_match_distance", config.tracking.max_match_distance));

    const cv::FileNode rendering = storage["rendering"];
    config.rendering.draw_debug_views = readBool(rendering, "draw_debug_views", config.rendering.draw_debug_views);
    config.rendering.draw_labels = readBool(rendering, "draw_labels", config.rendering.draw_labels);

    const cv::FileNode evaluation = storage["evaluation"];
    config.evaluation.manifest_path =
        resolvePathString(config_dir, readString(evaluation, "manifest_path", config.evaluation.manifest_path));
    config.evaluation.output_path =
        resolvePathString(config_dir, readString(evaluation, "output_path", config.evaluation.output_path));

    const cv::FileNode observability = storage["observability"];
    config.observability.artifact_root =
        resolvePathString(config_dir, readString(observability, "artifact_root", config.observability.artifact_root));
    config.observability.save_failure_samples =
        readBool(observability, "save_failure_samples", config.observability.save_failure_samples);
    config.observability.save_config_snapshot =
        readBool(observability, "save_config_snapshot", config.observability.save_config_snapshot);
    config.observability.log_per_frame = readBool(observability, "log_per_frame", config.observability.log_per_frame);

    return config;
}

std::unique_ptr<FrameSource> makeFrameSource(const AppConfig &config) {
    std::unique_ptr<FrameSource> source =
        (config.input.mode == InputMode::kWebcam) ? makeWebcamFrameSource() : makeBatchImageFrameSource();
    if (!source->open(config.input)) {
        throw std::runtime_error("Failed to open frame source");
    }
    return source;
}

std::unique_ptr<RunStorage> makeRunStorage(const AppConfig &config) {
    return makeSqliteRunStorage(config);
}

std::unique_ptr<Pipeline> makeDefaultPipeline(const AppConfig &config, RunStorage &storage) {
    return std::make_unique<Pipeline>(config, makeColorScorePreprocessor(config.preprocess),
                                      makeKMeansSegmenter(config.segmentation),
                                      makeBasicMorphologyProcessor(config.morphology),
                                      makeConnectedComponentAnalyzer(config.region), makeShapeFeatureExtractor(),
                                      makeNearestNeighborClassifier(config.classification, storage),
                                      makeCentroidTracker(config.tracking), makeOpenCvRenderer(config.rendering));
}

}  // namespace vision
