#pragma once

#include "vision/pipeline.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace vision {

struct RunMetrics {
    int processed_frames = 0;
    int frames_with_detection = 0;
    int unknown_predictions = 0;
    int labeled_frames = 0;
    int correct_predictions = 0;
    int exception_count = 0;
    double total_latency_ms = 0.0;
    double failed_stage_latency_ms = 0.0;
    double preprocess_ms_total = 0.0;
    double segmentation_ms_total = 0.0;
    double morphology_ms_total = 0.0;
    double components_ms_total = 0.0;
    double feature_ms_total = 0.0;
    double classification_ms_total = 0.0;
    double tracking_ms_total = 0.0;
    double rendering_ms_total = 0.0;
};

std::string joinCommandLine(int argc, char **argv);
std::string detectGitCommit();
std::string isoTimestampUtc();
std::string isoTimestampCompactUtc();
std::string sanitizeArtifactToken(std::string value);

void updateRunMetrics(RunMetrics &metrics, const PipelineResult &result, const std::string &predicted_label,
                      const std::string &true_label = std::string());
void recordRunException(RunMetrics &metrics);

class RunObserver {
public:
    RunObserver(const std::string &app_name, const AppConfig &config, const std::string &config_path,
                const std::string &command_line);

    const std::string &gitCommit() const;
    const std::string &startedAt() const;
    const std::filesystem::path &runDirectory() const;
    const std::filesystem::path &logPath() const;
    const std::filesystem::path &metricsPath() const;
    const std::filesystem::path &configSnapshotPath() const;
    const std::filesystem::path &failureDirectory() const;

    void logMessage(const std::string &level, const std::string &message);
    void logFrame(const FramePacket &frame, const PipelineResult &result, const std::string &true_label = std::string(),
                  const std::string &reason = std::string());
    void logException(const FramePacket &frame, const std::string &reason);
    bool saveFailureFrame(const FramePacket &frame, const PipelineResult &result, const std::string &true_label,
                          const std::string &predicted_label) const;
    bool writeMetrics(const RunMetrics &metrics, double wall_seconds,
                      const std::filesystem::path &output_path = std::filesystem::path()) const;

private:
    void writeConfigSnapshot(const AppConfig &config, const std::string &config_path, const std::string &command_line) const;

    std::string app_name_;
    std::string git_commit_;
    std::string started_at_;
    std::filesystem::path artifact_root_;
    std::filesystem::path run_directory_;
    std::filesystem::path log_path_;
    std::filesystem::path metrics_path_;
    std::filesystem::path config_snapshot_path_;
    std::filesystem::path failure_directory_;
    bool save_failure_samples_ = true;
    bool log_per_frame_ = true;
    mutable std::ofstream log_stream_;
};

}  // namespace vision
