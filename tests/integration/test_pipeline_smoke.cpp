#include "test_support.hpp"

#include <opencv2/imgproc.hpp>

#include <memory>

int main() {
    const auto root = vision_test::tempRoot("integration_pipeline_smoke");
    vision::AppConfig config;
    config.region.min_area = 100;
    config.region.max_regions = 1;
    config.storage.database_path = (root / "vision.sqlite3").string();
    config.storage.sample_image_dir = (root / "samples").string();
    config.evaluation.output_path = (root / "eval.csv").string();
    config.observability.artifact_root = (root / "artifacts").string();

    std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);
    std::unique_ptr<vision::Pipeline> pipeline = vision::makeDefaultPipeline(config, *storage);

    vision::FramePacket frame;
    frame.source_id = "synthetic";
    frame.frame_index = 0;
    frame.frame = cv::Mat(240, 320, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::rectangle(frame.frame, cv::Rect(100, 80, 80, 60), cv::Scalar(0, 0, 0), cv::FILLED);

    const vision::PipelineResult result = pipeline->run(frame);
    vision_test::require(!result.candidates.empty(), "The smoke frame should produce a detection.");
    vision_test::require(result.candidates.front().detection.track_id > 0, "The detection should receive a track ID.");
    vision_test::require(!result.artifacts.overlay.empty(), "The pipeline should render an overlay.");
    vision_test::require(!result.artifacts.binary_mask.empty(), "The pipeline should produce a binary mask.");
    return 0;
}
