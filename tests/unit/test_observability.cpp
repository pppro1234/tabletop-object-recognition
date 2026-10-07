#include "test_support.hpp"
#include "vision/observability.hpp"

#include <filesystem>
#include <fstream>

namespace {

namespace fs = std::filesystem;

std::string slurp(const fs::path &path) {
    std::ifstream input(path);
    vision_test::require(input.is_open(), "Failed to open file: " + path.string());
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
    const vision::AppConfig config = vision_test::makeTestConfig("unit_observability");
    vision::RunMetrics metrics;

    {
        vision::RunObserver observer("test_app", config, "configs/default.yaml", "./test_app --config configs/default.yaml");

        vision::FramePacket frame = vision_test::makeFrame("fixture.jpg", 7, vision_test::makeCanvas());
        vision::PipelineResult result;
        result.timings.preprocess_ms = 1.5;
        result.timings.segmentation_ms = 2.5;
        result.timings.morphology_ms = 0.5;
        result.timings.components_ms = 1.0;
        result.timings.feature_ms = 0.25;
        result.timings.classification_ms = 0.4;
        result.timings.tracking_ms = 0.1;
        result.timings.rendering_ms = 0.75;
        result.timings.total_ms = 7.0;

        vision::DetectionCandidate candidate;
        candidate.detection.label = "plate";
        candidate.detection.score = 0.9f;
        result.candidates.push_back(candidate);

        observer.logFrame(frame, result, "plate");
        vision::updateRunMetrics(metrics, result, "plate", "plate");

        result.candidates.front().detection.label = "Unknown";
        observer.logFrame(frame, result, "rod", "unknown_reject");
        vision::updateRunMetrics(metrics, result, "Unknown", "rod");

        vision_test::require(observer.saveFailureFrame(frame, result, "rod", "Unknown"),
                             "Failure frame should be written for observability.");
        vision_test::require(observer.writeMetrics(metrics, 0.5), "Metrics JSON should be written.");
    }

    const fs::path artifact_root = fs::path(config.observability.artifact_root);
    const fs::path runs_root = artifact_root / "runs";
    vision_test::require(fs::exists(runs_root), "Run artifact directory should exist.");

    fs::path run_dir;
    for (const fs::directory_entry &entry : fs::directory_iterator(runs_root)) {
        if (entry.is_directory()) {
            run_dir = entry.path();
            break;
        }
    }
    vision_test::require(!run_dir.empty(), "A run directory should be created.");
    vision_test::require(fs::exists(run_dir / "run.log"), "Per-run log file should exist.");
    vision_test::require(fs::exists(run_dir / "metrics.json"), "Per-run metrics file should exist.");
    vision_test::require(fs::exists(run_dir / "config_snapshot.yaml"), "Config snapshot should exist.");

    const std::string metrics_json = slurp(run_dir / "metrics.json");
    vision_test::require(metrics_json.find("\"avg_fps\"") != std::string::npos, "Metrics should contain FPS.");
    vision_test::require(metrics_json.find("\"unknown_reject_rate\"") != std::string::npos,
                         "Metrics should contain unknown reject rate.");
    vision_test::require(metrics_json.find("\"segmentation\"") != std::string::npos,
                         "Metrics should contain stage breakdown.");

    const std::string snapshot = slurp(run_dir / "config_snapshot.yaml");
    vision_test::require(snapshot.find("git_commit") != std::string::npos, "Snapshot should record git commit.");
    vision_test::require(snapshot.find("artifact_root") != std::string::npos,
                         "Snapshot should include observability config.");

    const fs::path failures_root = artifact_root / "failures";
    bool found_failure = false;
    for (const fs::directory_entry &entry : fs::directory_iterator(failures_root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            found_failure = true;
            break;
        }
    }
    vision_test::require(found_failure, "Failure artifacts should include a saved frame.");
    return 0;
}
