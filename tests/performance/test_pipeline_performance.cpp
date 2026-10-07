#include "test_support.hpp"
#include "vision/pipeline.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <vector>

namespace {

namespace fs = std::filesystem;

void trainFromImage(vision::Pipeline &pipeline, const fs::path &path, const std::string &label, std::int64_t frame_index) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    vision_test::require(!image.empty(), "Failed to load performance training image: " + path.string());

    const vision::FramePacket frame = vision_test::makeFrame(path.string(), frame_index, image);
    const vision::PipelineResult result = pipeline.run(frame);
    vision_test::require(!result.candidates.empty(), "Performance training fixture should yield a detection.");
    vision_test::require(pipeline.storeTrainingSample(frame, result, 0, label), "Failed to persist performance sample.");
}

struct StageStats {
    std::vector<double> total_ms;
    std::vector<double> preprocess_ms;
    std::vector<double> segmentation_ms;
    std::vector<double> morphology_ms;
    std::vector<double> components_ms;
    std::vector<double> feature_ms;
    std::vector<double> classification_ms;
    std::vector<double> tracking_ms;
    std::vector<double> rendering_ms;
};

}  // namespace

int main() {
    const fs::path golden_root = vision_test::sourceDir() / "tests/regression/golden";
    const std::vector<vision_test::ExpectedDetection> expectations =
        vision_test::loadExpectedDetections(golden_root / "expected_eval.csv");

    vision::AppConfig config = vision_test::makeTestConfig("performance_pipeline");
    std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);
    std::unique_ptr<vision::Pipeline> pipeline = vision::makeDefaultPipeline(config, *storage);

    trainFromImage(*pipeline, golden_root / "train/plate_train_a.ppm", "plate", 0);
    trainFromImage(*pipeline, golden_root / "train/plate_train_b.ppm", "plate", 1);
    trainFromImage(*pipeline, golden_root / "train/rod_train_a.ppm", "rod", 2);
    trainFromImage(*pipeline, golden_root / "train/rod_train_b.ppm", "rod", 3);

    std::vector<cv::Mat> dataset;
    dataset.reserve(expectations.size());
    for (const vision_test::ExpectedDetection &expected : expectations) {
        const cv::Mat image = cv::imread((golden_root / expected.image).string(), cv::IMREAD_COLOR);
        vision_test::require(!image.empty(), "Failed to load performance fixture: " + expected.image);
        dataset.push_back(image);
    }

    for (std::size_t index = 0; index < dataset.size(); ++index) {
        const vision::FramePacket warmup = vision_test::makeFrame("warmup", static_cast<std::int64_t>(index), dataset[index]);
        (void)pipeline->run(warmup);
    }

    constexpr int kLoops = 25;
    StageStats stats;
    const auto wall_start = std::chrono::steady_clock::now();
    for (int loop = 0; loop < kLoops; ++loop) {
        for (std::size_t index = 0; index < dataset.size(); ++index) {
            const vision::FramePacket frame =
                vision_test::makeFrame("perf_" + std::to_string(loop), static_cast<std::int64_t>(loop * 100 + index), dataset[index]);
            const vision::PipelineResult result = pipeline->run(frame);
            vision_test::require(!result.candidates.empty(), "Performance fixture lost detection coverage.");

            stats.total_ms.push_back(result.timings.total_ms);
            stats.preprocess_ms.push_back(result.timings.preprocess_ms);
            stats.segmentation_ms.push_back(result.timings.segmentation_ms);
            stats.morphology_ms.push_back(result.timings.morphology_ms);
            stats.components_ms.push_back(result.timings.components_ms);
            stats.feature_ms.push_back(result.timings.feature_ms);
            stats.classification_ms.push_back(result.timings.classification_ms);
            stats.tracking_ms.push_back(result.timings.tracking_ms);
            stats.rendering_ms.push_back(result.timings.rendering_ms);
        }
    }
    const double wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    const double avg_fps = (wall_seconds > 0.0) ? (static_cast<double>(stats.total_ms.size()) / wall_seconds) : 0.0;
    const double p50_latency = vision_test::percentile(stats.total_ms, 0.50);
    const double p95_latency = vision_test::percentile(stats.total_ms, 0.95);

    const fs::path report_path = vision_test::tempRoot("performance_pipeline_report") / "metrics.json";
    std::ofstream report(report_path);
    vision_test::require(report.is_open(), "Failed to write performance report.");
    report << std::fixed << std::setprecision(3)
           << "{\n"
           << "  \"frames\": " << stats.total_ms.size() << ",\n"
           << "  \"avg_fps\": " << avg_fps << ",\n"
           << "  \"p50_latency_ms\": " << p50_latency << ",\n"
           << "  \"p95_latency_ms\": " << p95_latency << ",\n"
           << "  \"avg_stage_ms\": {\n"
           << "    \"preprocess\": " << vision_test::mean(stats.preprocess_ms) << ",\n"
           << "    \"segmentation\": " << vision_test::mean(stats.segmentation_ms) << ",\n"
           << "    \"morphology\": " << vision_test::mean(stats.morphology_ms) << ",\n"
           << "    \"components\": " << vision_test::mean(stats.components_ms) << ",\n"
           << "    \"feature\": " << vision_test::mean(stats.feature_ms) << ",\n"
           << "    \"classification\": " << vision_test::mean(stats.classification_ms) << ",\n"
           << "    \"tracking\": " << vision_test::mean(stats.tracking_ms) << ",\n"
           << "    \"rendering\": " << vision_test::mean(stats.rendering_ms) << "\n"
           << "  }\n"
           << "}\n";

    vision_test::require(avg_fps > 2.0, "Average FPS is below the minimum regression guardrail.");
    vision_test::require(p50_latency > 0.0, "P50 latency should be positive.");
    vision_test::require(p95_latency < 250.0, "P95 latency exceeded the performance guardrail.");
    vision_test::require(vision_test::mean(stats.preprocess_ms) >= 0.0, "Preprocess timing must be non-negative.");
    vision_test::require(vision_test::mean(stats.components_ms) > 0.0, "Connected-components timing should be measurable.");
    vision_test::require(vision_test::mean(stats.total_ms) >=
                             vision_test::mean(stats.preprocess_ms) + vision_test::mean(stats.segmentation_ms),
                         "Total latency should dominate the sum of early stages.");
    return 0;
}
