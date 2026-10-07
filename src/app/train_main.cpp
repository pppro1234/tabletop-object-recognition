#include "vision/observability.hpp"
#include "vision/pipeline.hpp"

#include "cli_utils.hpp"

#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

namespace {

void printUsage() {
    std::cout << "Usage: tabletop_recognition_train --label <name> [--source <webcam|camera_index|image_file|image_dir>] "
                 "[--config <path>] [--database <sqlite>] [--artifacts <dir>] [--sample-dir <dir>]\n"
              << "Example: ./tabletop_recognition_train --label mug --source webcam\n";
}

}  // namespace

int main(int argc, char **argv) {
    const vision::app::ParsedArgs args = vision::app::parseArgs(argc, argv);
    if (vision::app::hasOption(args, "help")) {
        printUsage();
        return 0;
    }

    try {
        const std::string config_path = vision::app::optionOr(args, "config", "configs/default.yaml");
        const std::string label = vision::app::requireOption(args, "label");
        vision::AppConfig config = vision::loadConfig(config_path);
        vision::app::applyStorageOverrides(config, args);
        vision::app::applySourceOverride(config, vision::app::optionOr(args, "source", ""));
        vision::RunObserver observer("tabletop_recognition_train", config, config_path, vision::joinCommandLine(argc, argv));
        std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);
        std::unique_ptr<vision::Pipeline> pipeline = vision::makeDefaultPipeline(config, *storage);
        std::unique_ptr<vision::FrameSource> source = vision::makeFrameSource(config);

        vision::RunMetrics metrics;
        int processed = 0;
        int saved = 0;
        int skipped = 0;
        vision::FramePacket frame;
        const auto started = std::chrono::steady_clock::now();
        if (config.input.mode == vision::InputMode::kWebcam) {
            std::cout << "tabletop_recognition_train webcam mode: press 's' to save the current primary detection, 'q' to quit.\n";
            while (source->next(frame)) {
                ++processed;
                try {
                    const vision::PipelineResult result = pipeline->run(frame);
                    const std::string predicted_label =
                        result.candidates.empty() ? "Unknown" : result.candidates.front().detection.label;
                    const std::string reason = result.candidates.empty() ? "no_detection" : "";
                    observer.logFrame(frame, result, "", reason);
                    vision::updateRunMetrics(metrics, result, predicted_label);

                    if (!result.artifacts.overlay.empty()) {
                        cv::imshow("train", result.artifacts.overlay);
                    }

                    const int key = cv::waitKey(1) & 0xFF;
                    if (key == 27 || key == 'q' || key == 'Q') {
                        break;
                    }
                    if (key != 's' && key != 'S') {
                        continue;
                    }

                    if (result.candidates.empty()) {
                        ++skipped;
                        observer.logMessage("warn", "sample_skipped reason=no_detection frame_index=" +
                                                        std::to_string(frame.frame_index));
                        std::cout << "No detection to save on this frame.\n";
                        continue;
                    }

                    std::string save_error;
                    if (pipeline->storeTrainingSample(frame, result, 0, label, &save_error)) {
                        ++saved;
                        observer.logMessage("info", "sample_saved label=" + label + " frame_index=" +
                                                       std::to_string(frame.frame_index));
                        std::cout << "Saved sample " << saved << " for label=" << label << '\n';
                    } else {
                        ++skipped;
                        observer.logMessage("error", "sample_save_failed label=" + label + " frame_index=" +
                                                        std::to_string(frame.frame_index) + " detail=" + save_error);
                        std::cerr << "Failed to save sample: " << save_error << '\n';
                    }
                } catch (const std::exception &error) {
                    observer.logException(frame, error.what());
                    vision::recordRunException(metrics);
                    ++skipped;
                    continue;
                }
            }
        } else {
            while (source->next(frame)) {
                ++processed;
                try {
                    const vision::PipelineResult result = pipeline->run(frame);
                    const std::string predicted_label =
                        result.candidates.empty() ? "Unknown" : result.candidates.front().detection.label;
                    const std::string reason = result.candidates.empty() ? "no_detection" : "";
                    observer.logFrame(frame, result, "", reason);
                    vision::updateRunMetrics(metrics, result, predicted_label);

                    if (result.candidates.empty()) {
                        ++skipped;
                        continue;
                    }

                    std::string save_error;
                    if (pipeline->storeTrainingSample(frame, result, 0, label, &save_error)) {
                        ++saved;
                    } else {
                        ++skipped;
                        observer.logMessage("error", "sample_save_failed label=" + label + " frame_index=" +
                                                        std::to_string(frame.frame_index) + " detail=" + save_error);
                        std::cerr << "Failed to save sample: " << save_error << '\n';
                    }
                } catch (const std::exception &error) {
                    observer.logException(frame, error.what());
                    vision::recordRunException(metrics);
                    ++skipped;
                    continue;
                }
            }
        }

        source->close();
        const double elapsed_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (!observer.writeMetrics(metrics, elapsed_seconds)) {
            std::cerr << "Failed to write training metrics to " << observer.metricsPath() << '\n';
        }
        observer.logMessage("info", "run_finished processed_frames=" + std::to_string(metrics.processed_frames) +
                                        " saved_samples=" + std::to_string(saved));
        std::cout << "Processed " << processed << " frames, saved " << saved << " samples, skipped " << skipped
                  << ". classifier_db=" << pipeline->classifier().sampleCount() << " database=" << config.storage.database_path
                  << " sample_images=" << config.storage.sample_image_dir << " artifacts=" << observer.runDirectory().string()
                  << " git_commit=" << observer.gitCommit() << '\n';
        return (saved > 0) ? 0 : 2;
    } catch (const std::exception &error) {
        std::cerr << "tabletop_recognition_train failed: " << error.what() << '\n';
        printUsage();
        return 1;
    }
}
