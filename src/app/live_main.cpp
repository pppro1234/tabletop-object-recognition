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
    std::cout << "Usage: tabletop_recognition_live [--config <path>] [--source <webcam|camera_index|image_dir>]\n"
              << "Example: ./tabletop_recognition_live --config configs/webcam.yaml\n";
}

}  // namespace

int main(int argc, char **argv) {
    const vision::app::ParsedArgs args = vision::app::parseArgs(argc, argv);
    if (vision::app::hasOption(args, "help")) {
        printUsage();
        return 0;
    }

    try {
        const std::string config_path = vision::app::optionOr(args, "config", "configs/webcam.yaml");
        vision::AppConfig config = vision::loadConfig(config_path);
        vision::app::applySourceOverride(config, vision::app::optionOr(args, "source", ""));
        vision::RunObserver observer("tabletop_recognition_live", config, config_path, vision::joinCommandLine(argc, argv));
        std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);
        std::unique_ptr<vision::Pipeline> pipeline = vision::makeDefaultPipeline(config, *storage);
        std::unique_ptr<vision::FrameSource> source = vision::makeFrameSource(config);

        vision::RunMetrics metrics;
        vision::FramePacket frame;
        const auto started = std::chrono::steady_clock::now();
        while (source->next(frame)) {
            try {
                const vision::PipelineResult result = pipeline->run(frame);
                const std::string predicted_label =
                    result.candidates.empty() ? "Unknown" : result.candidates.front().detection.label;
                const std::string reason = result.candidates.empty() ? "no_detection" : "";
                observer.logFrame(frame, result, "", reason);
                vision::updateRunMetrics(metrics, result, predicted_label);

                if (!result.artifacts.overlay.empty()) {
                    cv::imshow("live", result.artifacts.overlay);
                }
                if (config.rendering.draw_debug_views) {
                    if (!result.artifacts.score_image.empty()) {
                        cv::imshow("score", result.artifacts.score_image);
                    }
                    if (!result.artifacts.binary_mask.empty()) {
                        cv::imshow("binary", result.artifacts.binary_mask);
                    }
                    if (!result.artifacts.cleaned_mask.empty()) {
                        cv::imshow("cleaned", result.artifacts.cleaned_mask);
                    }
                    if (!result.artifacts.region_map.empty()) {
                        cv::imshow("regions", result.artifacts.region_map);
                    }
                }
            } catch (const std::exception &error) {
                observer.logException(frame, error.what());
                vision::recordRunException(metrics);
                continue;
            }

            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        source->close();
        const double elapsed_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (!observer.writeMetrics(metrics, elapsed_seconds)) {
            std::cerr << "Failed to write live metrics to " << observer.metricsPath() << '\n';
        }
        observer.logMessage("info", "run_finished processed_frames=" + std::to_string(metrics.processed_frames));
        std::cout << "tabletop_recognition_live artifacts=" << observer.runDirectory().string() << " log=" << observer.logPath().string()
                  << " metrics=" << observer.metricsPath().string() << " git_commit=" << observer.gitCommit() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "tabletop_recognition_live failed: " << error.what() << '\n';
        printUsage();
        return 1;
    }
}
