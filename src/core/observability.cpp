#include "vision/observability.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>

namespace vision {
namespace {

namespace fs = std::filesystem;

std::string trimCopy(const std::string &value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string yamlQuote(const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"' || ch == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

const char *inputModeName(InputMode mode) {
    return (mode == InputMode::kBatchImages) ? "batch_images" : "webcam";
}

double safeDiv(double numerator, double denominator) {
    return (denominator > 0.0) ? (numerator / denominator) : 0.0;
}

void writeString(std::ostream &output, int indent, const std::string &key, const std::string &value) {
    output << std::string(static_cast<std::size_t>(indent), ' ') << key << ": " << yamlQuote(value) << '\n';
}

void writeNumber(std::ostream &output, int indent, const std::string &key, double value) {
    output << std::string(static_cast<std::size_t>(indent), ' ') << key << ": " << value << '\n';
}

void writeInteger(std::ostream &output, int indent, const std::string &key, int value) {
    output << std::string(static_cast<std::size_t>(indent), ' ') << key << ": " << value << '\n';
}

void writeBool(std::ostream &output, int indent, const std::string &key, bool value) {
    output << std::string(static_cast<std::size_t>(indent), ' ') << key << ": " << (value ? 1 : 0) << '\n';
}

std::string detectionLabel(const PipelineResult &result) {
    return result.candidates.empty() ? "NoDetection" : result.candidates.front().detection.label;
}

double detectionScore(const PipelineResult &result) {
    return result.candidates.empty() ? 0.0 : result.candidates.front().detection.score;
}

}  // namespace

std::string joinCommandLine(int argc, char **argv) {
    std::ostringstream stream;
    for (int i = 0; i < argc; ++i) {
        if (i > 0) {
            stream << ' ';
        }
        stream << argv[i];
    }
    return stream.str();
}

std::string detectGitCommit() {
    std::string output;
    FILE *pipe = popen("git rev-parse --verify HEAD 2>/dev/null", "r");
    if (pipe == nullptr) {
        return "nogit";
    }

    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    pclose(pipe);
    output = trimCopy(output);
    if (output.size() < 7) {
        return "nogit";
    }
    for (char ch : output) {
        if (!std::isxdigit(static_cast<unsigned char>(ch))) {
            return "nogit";
        }
    }
    return output;
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

std::string isoTimestampCompactUtc() {
    const std::time_t raw_time = std::time(nullptr);
    std::tm time_info{};
#if defined(_WIN32)
    gmtime_s(&time_info, &raw_time);
#else
    gmtime_r(&raw_time, &time_info);
#endif

    std::ostringstream stream;
    stream << std::put_time(&time_info, "%Y%m%dT%H%M%SZ");
    return stream.str();
}

std::string sanitizeArtifactToken(std::string value) {
    for (char &ch : value) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return value.empty() ? "item" : value;
}

void updateRunMetrics(RunMetrics &metrics, const PipelineResult &result, const std::string &predicted_label,
                      const std::string &true_label) {
    ++metrics.processed_frames;
    if (!result.candidates.empty()) {
        ++metrics.frames_with_detection;
        if (predicted_label == "Unknown") {
            ++metrics.unknown_predictions;
        }
    }
    if (!true_label.empty()) {
        ++metrics.labeled_frames;
        if (predicted_label == true_label) {
            ++metrics.correct_predictions;
        }
    }

    metrics.total_latency_ms += result.timings.total_ms;
    metrics.preprocess_ms_total += result.timings.preprocess_ms;
    metrics.segmentation_ms_total += result.timings.segmentation_ms;
    metrics.morphology_ms_total += result.timings.morphology_ms;
    metrics.components_ms_total += result.timings.components_ms;
    metrics.feature_ms_total += result.timings.feature_ms;
    metrics.classification_ms_total += result.timings.classification_ms;
    metrics.tracking_ms_total += result.timings.tracking_ms;
    metrics.rendering_ms_total += result.timings.rendering_ms;
}

void recordRunException(RunMetrics &metrics) {
    ++metrics.processed_frames;
    ++metrics.exception_count;
}

RunObserver::RunObserver(const std::string &app_name, const AppConfig &config, const std::string &config_path,
                         const std::string &command_line)
    : app_name_(app_name),
      git_commit_(detectGitCommit()),
      started_at_(isoTimestampUtc()),
      artifact_root_(config.observability.artifact_root),
      save_failure_samples_(config.observability.save_failure_samples),
      log_per_frame_(config.observability.log_per_frame) {
    const std::string run_token =
        sanitizeArtifactToken(app_name_ + "_" + config.run_name + "_" + isoTimestampCompactUtc());
    run_directory_ = artifact_root_ / "runs" / run_token;
    log_path_ = run_directory_ / "run.log";
    metrics_path_ = run_directory_ / "metrics.json";
    config_snapshot_path_ = run_directory_ / "config_snapshot.yaml";
    failure_directory_ = artifact_root_ / "failures";

    std::error_code ec;
    fs::create_directories(run_directory_, ec);
    fs::create_directories(failure_directory_, ec);

    log_stream_.open(log_path_, std::ios::app);
    if (!log_stream_.is_open()) {
        throw std::runtime_error("Failed to open log file: " + log_path_.string());
    }

    if (config.observability.save_config_snapshot) {
        writeConfigSnapshot(config, config_path, command_line);
    }

    logMessage("info", "run_started app=" + app_name_ + " git_commit=" + git_commit_ + " run_dir=" + run_directory_.string());
}

const std::string &RunObserver::gitCommit() const {
    return git_commit_;
}

const std::string &RunObserver::startedAt() const {
    return started_at_;
}

const fs::path &RunObserver::runDirectory() const {
    return run_directory_;
}

const fs::path &RunObserver::logPath() const {
    return log_path_;
}

const fs::path &RunObserver::metricsPath() const {
    return metrics_path_;
}

const fs::path &RunObserver::configSnapshotPath() const {
    return config_snapshot_path_;
}

const fs::path &RunObserver::failureDirectory() const {
    return failure_directory_;
}

void RunObserver::logMessage(const std::string &level, const std::string &message) {
    log_stream_ << "[" << isoTimestampUtc() << "] level=" << level << " " << message << '\n';
    log_stream_.flush();
}

void RunObserver::logFrame(const FramePacket &frame, const PipelineResult &result, const std::string &true_label,
                           const std::string &reason) {
    if (!log_per_frame_) {
        return;
    }

    log_stream_ << "[" << isoTimestampUtc() << "] level=info event=frame"
                << " frame_index=" << frame.frame_index << " source=" << yamlQuote(frame.source_id)
                << " detected=" << (result.candidates.empty() ? 0 : 1) << " predicted_label=" << yamlQuote(detectionLabel(result))
                << " score=" << detectionScore(result) << " true_label=" << yamlQuote(true_label)
                << " latency_ms=" << result.timings.total_ms << " preprocess_ms=" << result.timings.preprocess_ms
                << " segmentation_ms=" << result.timings.segmentation_ms
                << " morphology_ms=" << result.timings.morphology_ms
                << " components_ms=" << result.timings.components_ms << " feature_ms=" << result.timings.feature_ms
                << " classification_ms=" << result.timings.classification_ms << " tracking_ms=" << result.timings.tracking_ms
                << " rendering_ms=" << result.timings.rendering_ms << " reason=" << yamlQuote(reason) << '\n';
    log_stream_.flush();
}

void RunObserver::logException(const FramePacket &frame, const std::string &reason) {
    log_stream_ << "[" << isoTimestampUtc() << "] level=error event=frame_exception"
                << " frame_index=" << frame.frame_index << " source=" << yamlQuote(frame.source_id)
                << " reason=" << yamlQuote(reason) << '\n';
    log_stream_.flush();
}

bool RunObserver::saveFailureFrame(const FramePacket &frame, const PipelineResult &result, const std::string &true_label,
                                   const std::string &predicted_label) const {
    if (!save_failure_samples_ || frame.frame.empty()) {
        return false;
    }

    const cv::Mat output = !result.artifacts.overlay.empty() ? result.artifacts.overlay : frame.frame;
    const std::string filename = sanitizeArtifactToken(true_label) + "_" + sanitizeArtifactToken(predicted_label) + "_" +
                                 isoTimestampCompactUtc() + "_" + std::to_string(frame.frame_index) + ".png";
    const fs::path output_path = failure_directory_ / filename;
    return cv::imwrite(output_path.string(), output);
}

bool RunObserver::writeMetrics(const RunMetrics &metrics, double wall_seconds, const fs::path &output_path) const {
    const fs::path destination = output_path.empty() ? metrics_path_ : output_path;
    std::error_code ec;
    fs::create_directories(destination.parent_path(), ec);

    std::ofstream output(destination);
    if (!output.is_open()) {
        return false;
    }

    const double frames = static_cast<double>(metrics.processed_frames);
    const double completed_frames = static_cast<double>(metrics.processed_frames - metrics.exception_count);
    const double avg_fps = safeDiv(frames, wall_seconds);
    const double accuracy = safeDiv(static_cast<double>(metrics.correct_predictions), static_cast<double>(metrics.labeled_frames));
    const double unknown_reject_rate =
        safeDiv(static_cast<double>(metrics.unknown_predictions), static_cast<double>(metrics.frames_with_detection));

    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"app\": " << yamlQuote(app_name_) << ",\n"
           << "  \"started_at\": " << yamlQuote(started_at_) << ",\n"
           << "  \"git_commit\": " << yamlQuote(git_commit_) << ",\n"
           << "  \"processed_frames\": " << metrics.processed_frames << ",\n"
           << "  \"completed_pipeline_frames\": " << static_cast<int>(completed_frames) << ",\n"
           << "  \"frames_with_detection\": " << metrics.frames_with_detection << ",\n"
           << "  \"frames_without_detection\": " << std::max(0, metrics.processed_frames - metrics.exception_count - metrics.frames_with_detection)
           << ",\n"
           << "  \"labeled_frames\": " << metrics.labeled_frames << ",\n"
           << "  \"correct_predictions\": " << metrics.correct_predictions << ",\n"
           << "  \"exception_count\": " << metrics.exception_count << ",\n"
           << "  \"avg_fps\": " << avg_fps << ",\n"
           << "  \"accuracy\": " << accuracy << ",\n"
           << "  \"unknown_reject_rate\": " << unknown_reject_rate << ",\n"
           << "  \"avg_latency_ms\": " << safeDiv(metrics.total_latency_ms, completed_frames) << ",\n"
           << "  \"avg_failed_stage_ms\": " << safeDiv(metrics.failed_stage_latency_ms, metrics.exception_count) << ",\n"
           << "  \"avg_stage_ms\": {\n"
           << "    \"preprocess\": " << safeDiv(metrics.preprocess_ms_total, completed_frames) << ",\n"
           << "    \"segmentation\": " << safeDiv(metrics.segmentation_ms_total, completed_frames) << ",\n"
           << "    \"morphology\": " << safeDiv(metrics.morphology_ms_total, completed_frames) << ",\n"
           << "    \"components\": " << safeDiv(metrics.components_ms_total, completed_frames) << ",\n"
           << "    \"feature\": " << safeDiv(metrics.feature_ms_total, completed_frames) << ",\n"
           << "    \"classification\": " << safeDiv(metrics.classification_ms_total, completed_frames) << ",\n"
           << "    \"tracking\": " << safeDiv(metrics.tracking_ms_total, completed_frames) << ",\n"
           << "    \"rendering\": " << safeDiv(metrics.rendering_ms_total, completed_frames) << "\n"
           << "  }\n"
           << "}\n";
    return true;
}

void RunObserver::writeConfigSnapshot(const AppConfig &config, const std::string &config_path,
                                      const std::string &command_line) const {
    std::ofstream output(config_snapshot_path_);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open config snapshot: " + config_snapshot_path_.string());
    }

    output << "%YAML:1.0\n";
    output << "runtime_metadata:\n";
    writeString(output, 2, "app_name", app_name_);
    writeString(output, 2, "started_at", started_at_);
    writeString(output, 2, "git_commit", git_commit_);
    writeString(output, 2, "config_path", config_path);
    writeString(output, 2, "command_line", command_line);
    writeString(output, 0, "run_name", config.run_name);

    if (config.labels.empty()) {
        output << "labels: []\n";
    } else {
        output << "labels:\n";
        for (const std::string &label : config.labels) {
            output << "  - " << yamlQuote(label) << '\n';
        }
    }

    output << "input:\n";
    writeString(output, 2, "mode", inputModeName(config.input.mode));
    writeInteger(output, 2, "webcam_index", config.input.webcam_index);
    writeString(output, 2, "image_dir", config.input.image_dir);
    writeString(output, 2, "image_glob", config.input.image_glob);

    output << "preprocess:\n";
    writeInteger(output, 2, "blur_kernel_size", config.preprocess.blur_kernel_size);
    writeInteger(output, 2, "saturation_weight", config.preprocess.saturation_weight);

    output << "segmentation:\n";
    writeInteger(output, 2, "threshold_offset", config.segmentation.threshold_offset);
    writeInteger(output, 2, "kmeans_iterations", config.segmentation.kmeans_iterations);
    writeInteger(output, 2, "sampling_stride", config.segmentation.sampling_stride);

    output << "morphology:\n";
    writeInteger(output, 2, "kernel_size", config.morphology.kernel_size);
    writeInteger(output, 2, "open_iterations", config.morphology.open_iterations);
    writeInteger(output, 2, "close_iterations", config.morphology.close_iterations);

    output << "region:\n";
    writeInteger(output, 2, "min_area", config.region.min_area);
    writeInteger(output, 2, "max_regions", config.region.max_regions);
    writeBool(output, 2, "reject_boundary", config.region.reject_boundary);

    output << "classification:\n";
    writeNumber(output, 2, "unknown_threshold", config.classification.unknown_threshold);

    output << "storage:\n";
    writeString(output, 2, "database_path", config.storage.database_path);
    writeString(output, 2, "sample_image_dir", config.storage.sample_image_dir);

    output << "tracking:\n";
    writeNumber(output, 2, "max_match_distance", config.tracking.max_match_distance);

    output << "rendering:\n";
    writeBool(output, 2, "draw_debug_views", config.rendering.draw_debug_views);
    writeBool(output, 2, "draw_labels", config.rendering.draw_labels);

    output << "evaluation:\n";
    writeString(output, 2, "manifest_path", config.evaluation.manifest_path);
    writeString(output, 2, "output_path", config.evaluation.output_path);

    output << "observability:\n";
    writeString(output, 2, "artifact_root", config.observability.artifact_root);
    writeBool(output, 2, "save_failure_samples", config.observability.save_failure_samples);
    writeBool(output, 2, "save_config_snapshot", config.observability.save_config_snapshot);
    writeBool(output, 2, "log_per_frame", config.observability.log_per_frame);
}

}  // namespace vision
