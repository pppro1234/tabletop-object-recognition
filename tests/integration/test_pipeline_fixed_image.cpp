#include "test_support.hpp"
#include "vision/pipeline.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <memory>

namespace {

namespace fs = std::filesystem;

void trainFromImage(vision::Pipeline &pipeline, const fs::path &path, const std::string &label, std::int64_t frame_index) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    vision_test::require(!image.empty(), "Failed to load training image: " + path.string());

    const vision::FramePacket frame = vision_test::makeFrame(path.string(), frame_index, image);
    const vision::PipelineResult result = pipeline.run(frame);
    vision_test::require(!result.candidates.empty(), "Training image should produce at least one detection.");
    vision_test::require(pipeline.storeTrainingSample(frame, result, 0, label), "Failed to store training sample.");
}

}  // namespace

int main() {
    const fs::path golden_root = vision_test::sourceDir() / "tests/regression/golden";
    const std::vector<vision_test::ExpectedDetection> expected =
        vision_test::loadExpectedDetections(golden_root / "expected_eval.csv");

    vision::AppConfig config = vision_test::makeTestConfig("integration_pipeline_fixed_image");
    std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);
    std::unique_ptr<vision::Pipeline> pipeline = vision::makeDefaultPipeline(config, *storage);

    trainFromImage(*pipeline, golden_root / "train/plate_train_a.ppm", "plate", 0);
    trainFromImage(*pipeline, golden_root / "train/plate_train_b.ppm", "plate", 1);
    trainFromImage(*pipeline, golden_root / "train/rod_train_a.ppm", "rod", 2);
    trainFromImage(*pipeline, golden_root / "train/rod_train_b.ppm", "rod", 3);

    const vision_test::ExpectedDetection &target = expected.front();
    const fs::path eval_path = golden_root / target.image;
    const cv::Mat image = cv::imread(eval_path.string(), cv::IMREAD_COLOR);
    vision_test::require(!image.empty(), "Failed to load integration fixture image.");

    const vision::FramePacket frame = vision_test::makeFrame(eval_path.string(), 4, image);
    const vision::PipelineResult result = pipeline->run(frame);
    vision_test::require(!result.candidates.empty(), "Pipeline should detect the target on the fixed integration image.");

    const vision::Detection &detection = result.candidates.front().detection;
    vision_test::require(detection.label == target.label, "Integrated pipeline should keep the expected class label.");
    vision_test::requireNear(detection.centroid.x, target.centroid_x, 4.0, "Centroid x drift exceeds tolerance.");
    vision_test::requireNear(detection.centroid.y, target.centroid_y, 4.0, "Centroid y drift exceeds tolerance.");
    vision_test::require(vision_test::angleDistanceMod180(detection.angle_deg, target.angle_deg) <= 8.0,
                         "Angle drift exceeds integration tolerance.");
    vision_test::require(vision_test::intersectionOverUnion(detection.bounding_box, target.bbox) >= 0.72,
                         "Bounding box IoU should stay within the accepted integration tolerance.");
    return 0;
}
