#include "test_support.hpp"
#include "vision/pipeline.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <memory>
#include <numeric>
#include <unordered_map>

namespace {

namespace fs = std::filesystem;

void trainFromImage(vision::Pipeline &pipeline, const fs::path &path, const std::string &label, std::int64_t frame_index) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    vision_test::require(!image.empty(), "Failed to load regression training image: " + path.string());

    const vision::FramePacket frame = vision_test::makeFrame(path.string(), frame_index, image);
    const vision::PipelineResult result = pipeline.run(frame);
    vision_test::require(!result.candidates.empty(), "Regression training fixture should yield a detection.");
    vision_test::require(pipeline.storeTrainingSample(frame, result, 0, label), "Failed to persist regression training sample.");
}

}  // namespace

int main() {
    const fs::path golden_root = vision_test::sourceDir() / "tests/regression/golden";
    const std::vector<vision_test::ExpectedDetection> expectations =
        vision_test::loadExpectedDetections(golden_root / "expected_eval.csv");

    vision::AppConfig config = vision_test::makeTestConfig("regression_pipeline_golden");
    std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);
    std::unique_ptr<vision::Pipeline> pipeline = vision::makeDefaultPipeline(config, *storage);

    trainFromImage(*pipeline, golden_root / "train/plate_train_a.ppm", "plate", 0);
    trainFromImage(*pipeline, golden_root / "train/plate_train_b.ppm", "plate", 1);
    trainFromImage(*pipeline, golden_root / "train/rod_train_a.ppm", "rod", 2);
    trainFromImage(*pipeline, golden_root / "train/rod_train_b.ppm", "rod", 3);

    int correct = 0;
    std::vector<double> centroid_errors;
    std::vector<double> angle_errors;
    std::vector<double> bbox_ious;

    for (std::size_t index = 0; index < expectations.size(); ++index) {
        const vision_test::ExpectedDetection &expected = expectations[index];
        const fs::path image_path = golden_root / expected.image;
        const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        vision_test::require(!image.empty(), "Failed to load regression evaluation image: " + image_path.string());

        const vision::FramePacket frame = vision_test::makeFrame(image_path.string(), static_cast<std::int64_t>(index + 10), image);
        const vision::PipelineResult result = pipeline->run(frame);
        vision_test::require(!result.candidates.empty(), "Regression image lost its primary detection: " + expected.image);

        const vision::Detection &detection = result.candidates.front().detection;
        if (detection.label == expected.label) {
            ++correct;
        }

        const double centroid_error =
            std::hypot(detection.centroid.x - expected.centroid_x, detection.centroid.y - expected.centroid_y);
        const double angle_error = vision_test::angleDistanceMod180(detection.angle_deg, expected.angle_deg);
        const double bbox_iou = vision_test::intersectionOverUnion(detection.bounding_box, expected.bbox);

        centroid_errors.push_back(centroid_error);
        angle_errors.push_back(angle_error);
        bbox_ious.push_back(bbox_iou);

        vision_test::require(detection.label == expected.label,
                             "Regression classification changed for fixture: " + expected.image + " expected=" +
                                 expected.label + " actual=" + detection.label);
        vision_test::require(centroid_error <= 5.0, "Regression centroid error exceeded tolerance for: " + expected.image);
        vision_test::require(angle_error <= 10.0, "Regression angle error exceeded tolerance for: " + expected.image);
        vision_test::require(bbox_iou >= 0.68, "Regression bounding box IoU dropped below tolerance for: " + expected.image);
    }

    const double accuracy = static_cast<double>(correct) / static_cast<double>(expectations.size());
    vision_test::requireNear(accuracy, 1.0, 1e-6, "Golden-set accuracy regressed.");
    vision_test::require(vision_test::mean(centroid_errors) <= 3.5, "Average centroid error regressed.");
    vision_test::require(vision_test::mean(angle_errors) <= 7.0, "Average angle error regressed.");
    vision_test::require(*std::min_element(bbox_ious.begin(), bbox_ious.end()) >= 0.68,
                         "Worst-case bounding box IoU regressed.");
    return 0;
}
