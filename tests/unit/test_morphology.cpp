#include "factories.hpp"
#include "test_support.hpp"

#include <opencv2/core.hpp>

int main() {
    vision::MorphologyConfig config;
    config.kernel_size = 3;
    config.open_iterations = 1;
    config.close_iterations = 1;

    std::unique_ptr<vision::MorphologyProcessor> morphology = vision::makeBasicMorphologyProcessor(config);

    cv::Mat mask = cv::Mat::zeros(25, 25, CV_8UC1);
    mask(cv::Rect(7, 7, 11, 11)).setTo(cv::Scalar(255));
    mask.at<unsigned char>(3, 3) = 255;
    mask.at<unsigned char>(12, 12) = 0;

    const cv::Mat cleaned = morphology->refine(mask);
    vision_test::require(!cleaned.empty(), "Morphology should return a mask.");
    vision_test::require(cleaned.at<unsigned char>(3, 3) == 0, "Opening should remove isolated noise pixels.");
    vision_test::require(cleaned.at<unsigned char>(12, 12) == 255, "Closing should fill small holes inside the target.");
    vision_test::require(cv::countNonZero(cleaned) >= 110, "The main component should remain after morphology cleanup.");
    return 0;
}
