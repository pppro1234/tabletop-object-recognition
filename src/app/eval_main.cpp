#include "vision/observability.hpp"
#include "vision/pipeline.hpp"

#include "cli_utils.hpp"
#include "evaluation_manifest.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void printUsage() {
    std::cout << "Usage: tabletop_recognition_eval [--config <path>] [--dataset <image_dir>] [--out <result_dir_or_csv>] "
                 "[--manifest <csv>] [--database <sqlite>] [--artifacts <dir>] [--sample-dir <dir>]\n"
              << "Manifest: image,label (one labeled image per row; relative paths are based on the manifest directory).\n"
              << "--dataset explicitly overrides the root for relative image paths.\n"
              << "Example: ./tabletop_recognition_eval --manifest data/test_set/manifest.csv --out results/run_001\n";
}

fs::path resolveEvalOutputPath(const std::string &out_value) {
    const fs::path path(out_value);
    if (path.empty()) {
        return {};
    }
    if (path.has_extension()) {
        return path;
    }
    return path / "eval_results.csv";
}

std::string datasetNameFromPath(const std::string &dataset_path) {
    const fs::path path(dataset_path);
    const std::string filename = path.filename().string();
    return filename.empty() ? dataset_path : filename;
}

}  // namespace

int main(int argc, char **argv) {
    const vision::app::ParsedArgs args = vision::app::parseArgs(argc, argv);
    if (vision::app::hasOption(args, "help")) {
        printUsage();
        return 0;
    }

    try {
        const std::string config_path = vision::app::optionOr(args, "config", "configs/eval.yaml");
        vision::AppConfig config = vision::loadConfig(config_path);
        vision::app::applyStorageOverrides(config, args);
        config.input.mode = vision::InputMode::kBatchImages;

        const std::string dataset_override = vision::app::hasOption(args, "dataset")
                                                 ? vision::app::requireOption(args, "dataset") : "";
        if (!dataset_override.empty()) {
            config.input.image_dir = dataset_override;
        }

        const std::string manifest_override = vision::app::hasOption(args, "manifest")
                                                  ? vision::app::requireOption(args, "manifest") : "";
        if (!manifest_override.empty()) {
            config.evaluation.manifest_path = manifest_override;
        }

        const std::string out_override = vision::app::hasOption(args, "out")
                                             ? vision::app::requireOption(args, "out") : "";
        fs::path metrics_output_path;
        fs::path confusion_output_path;
        if (!out_override.empty()) {
            const fs::path eval_output_path = resolveEvalOutputPath(out_override);
            config.evaluation.output_path = eval_output_path.string();
            metrics_output_path = eval_output_path.parent_path() / "metrics.json";
            confusion_output_path = eval_output_path.parent_path() / "confusion_matrix.csv";
        } else {
            metrics_output_path = fs::path(config.evaluation.output_path).parent_path() / "metrics.json";
            confusion_output_path = fs::path(config.evaluation.output_path).parent_path() / "confusion_matrix.csv";
        }

        const std::vector<vision::app::EvaluationInput> manifest =
            vision::app::loadEvaluationManifest(config.evaluation.manifest_path, dataset_override);
        config.evaluation.manifest_path = fs::absolute(config.evaluation.manifest_path).lexically_normal().string();
        config.input.image_dir = dataset_override.empty()
                                     ? fs::path(config.evaluation.manifest_path).parent_path().string()
                                     : fs::absolute(dataset_override).lexically_normal().string();

        vision::RunObserver observer("tabletop_recognition_eval", config, config_path, vision::joinCommandLine(argc, argv));
        std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);
        std::unique_ptr<vision::Pipeline> pipeline = vision::makeDefaultPipeline(config, *storage);
        vision::app::writeEvaluationManifestSnapshot(observer.runDirectory() / "eval_manifest.csv", manifest);

        vision::RunRecord run;
        run.config_name = config.run_name.empty() ? fs::path(config_path).filename().string() : config.run_name;
        run.git_commit = observer.gitCommit();
        run.dataset_name = datasetNameFromPath(config.input.image_dir);
        run.started_at = observer.startedAt();
        const std::int64_t run_id = storage->beginRun(run);
        if (run_id <= 0) {
            std::cerr << "Failed to register evaluation run in the database.\n";
            return 4;
        }

        std::vector<vision::EvaluationRecord> records;
        vision::RunMetrics metrics;
        int failed_inputs = 0;
        int artifact_errors = 0;
        const auto save_failure = [&](const vision::FramePacket &frame, const vision::PipelineResult &result,
                                      const std::string &true_label, const std::string &predicted_label) {
            if (!config.observability.save_failure_samples || frame.frame.empty()) {
                return;
            }
            try {
                if (!observer.saveFailureFrame(frame, result, true_label, predicted_label)) {
                    throw std::runtime_error("Failed to save failure image for " + frame.source_id);
                }
            } catch (const std::exception &error) {
                ++artifact_errors;
                observer.logMessage("error", "failure_artifact_error: " + std::string(error.what()));
                std::cerr << "failure_artifact_error: " << error.what() << '\n';
            }
        };
        const auto eval_start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < manifest.size(); ++index) {
            const vision::app::EvaluationInput &input = manifest[index];
            const std::string &true_label = input.label;
            vision::FramePacket frame;
            frame.source_id = input.image_path.string();
            frame.frame_index = static_cast<std::int64_t>(index);
            frame.timestamp_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - eval_start).count();
            auto stage_start = std::chrono::steady_clock::now();
            std::string error_status = "unreadable_image";
            std::string error_prediction = "UnreadableImage";
            try {
                std::error_code file_error;
                const bool exists = fs::exists(input.image_path, file_error);
                if (!exists && !file_error) {
                    error_status = "missing_image";
                    error_prediction = "MissingImage";
                    throw std::runtime_error("Manifest image does not exist: " + frame.source_id);
                }
                if (file_error || !fs::is_regular_file(input.image_path)) {
                    throw std::runtime_error("Manifest image is not a readable regular file: " + frame.source_id);
                }
                frame.frame = cv::imread(frame.source_id, cv::IMREAD_COLOR);
                if (frame.frame.empty()) {
                    throw std::runtime_error("Cannot decode manifest image: " + frame.source_id);
                }
                error_status = "pipeline_error";
                error_prediction = "PipelineError";
                stage_start = std::chrono::steady_clock::now();
                const vision::PipelineResult result = pipeline->run(frame);
                const bool has_detection = !result.candidates.empty();
                const std::string predicted_label = has_detection ? result.candidates.front().detection.label : "NoDetection";
                const std::string status = !has_detection ? "no_detection"
                                                         : (predicted_label == true_label ? "ok" : "misclassified");
                observer.logFrame(frame, result, true_label, status);
                vision::updateRunMetrics(metrics, result, predicted_label, true_label);

                vision::EvaluationRecord record;
                record.frame_id = frame.source_id;
                record.true_label = true_label;
                record.predicted_label = predicted_label;
                record.score = has_detection ? result.candidates.front().detection.score : 0.0f;
                record.latency_ms = result.timings.total_ms;
                record.correct = has_detection && predicted_label == true_label;
                record.status = status;
                record.detection_count = static_cast<int>(result.candidates.size());
                records.push_back(record);

                if (!record.correct) {
                    save_failure(frame, result, true_label, predicted_label);
                }
            } catch (const std::exception &error) {
                const double failed_stage_ms = std::chrono::duration<double, std::milli>(
                                                   std::chrono::steady_clock::now() - stage_start).count();
                observer.logException(frame, error_status + ": " + error.what());
                std::cerr << error_status << ": " << error.what() << '\n';
                vision::recordRunException(metrics);
                ++metrics.labeled_frames;
                metrics.failed_stage_latency_ms += failed_stage_ms;
                ++failed_inputs;

                vision::PipelineResult empty_result;
                save_failure(frame, empty_result, true_label, error_prediction);

                vision::EvaluationRecord record;
                record.frame_id = frame.source_id;
                record.true_label = true_label;
                record.predicted_label = error_prediction;
                record.latency_ms = failed_stage_ms;
                record.correct = false;
                record.status = error_status;
                record.detection_count = 0;
                records.push_back(record);
            }
        }

        const double elapsed_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_start).count();
        const double avg_fps =
            (elapsed_seconds > 0.0) ? (static_cast<double>(metrics.processed_frames) / elapsed_seconds) : 0.0;

        if (!storage->appendPredictions(run_id, records)) {
            std::cerr << "Failed to store prediction rows in the database.\n";
            return 5;
        }
        if (!storage->writeEvaluation(records)) {
            std::cerr << "Failed to write evaluation results.\n";
            return 2;
        }
        const double accuracy =
            (metrics.labeled_frames > 0) ? (100.0 * static_cast<double>(metrics.correct_predictions) / metrics.labeled_frames) : 0.0;
        if (!storage->finishRun(run_id, avg_fps, accuracy)) {
            std::cerr << "Failed to finalize run metrics in the database.\n";
            return 6;
        }
        if (!storage->exportConfusionMatrix(run_id, confusion_output_path.string())) {
            std::cerr << "Failed to export confusion matrix.\n";
            return 7;
        }
        if (!observer.writeMetrics(metrics, elapsed_seconds)) {
            std::cerr << "Failed to write run metrics to " << observer.metricsPath() << ".\n";
            return 8;
        }
        if (!observer.writeMetrics(metrics, elapsed_seconds, metrics_output_path)) {
            std::cerr << "Failed to write metrics summary.\n";
            return 3;
        }
        observer.logMessage("info", "run_finished run_id=" + std::to_string(run_id) + " processed_frames=" +
                                        std::to_string(metrics.processed_frames) + " input_errors=" + std::to_string(failed_inputs) +
                                        " artifact_errors=" + std::to_string(artifact_errors));

        std::cout << "Evaluated " << records.size() << " manifest entries. input_errors=" << failed_inputs
                  << " artifact_errors=" << artifact_errors << " labeled=" << metrics.labeled_frames << " accuracy=" << accuracy
                  << "% avg_fps=" << avg_fps << " run_id=" << run_id << " output=" << config.evaluation.output_path
                  << " metrics=" << metrics_output_path.string() << " confusion=" << confusion_output_path.string()
                  << " artifacts=" << observer.runDirectory().string() << " failures=" << observer.failureDirectory().string()
                  << " git_commit=" << observer.gitCommit() << '\n';
        return failed_inputs == 0 && artifact_errors == 0 ? 0 : 9;
    } catch (const std::exception &error) {
        std::cerr << "tabletop_recognition_eval failed: " << error.what() << '\n';
        printUsage();
        return 1;
    }
}
