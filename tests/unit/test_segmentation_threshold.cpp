#include "factories.hpp"
#include "test_support.hpp"

#include <opencv2/core.hpp>

int main() {
    vision::SegmentationConfig base_config;
    base_config.sampling_stride = 1;
    base_config.kmeans_iterations = 25;

    std::unique_ptr<vision::Segmenter> segmenter = vision::makeKMeansSegmenter(base_config);
    cv::Mat bimodal(20, 20, CV_8UC1, cv::Scalar(220));
    bimodal(cv::Rect(0, 0, 10, 20)).setTo(cv::Scalar(30));

    const cv::Mat mask = segmenter->segment(bimodal);
    vision_test::require(!mask.empty(), "Segmenter should return a mask for a valid grayscale score image.");
    vision_test::require(cv::countNonZero(mask(cv::Rect(0, 0, 10, 20))) == 200,
                         "Dark cluster should be classified as foreground.");
    vision_test::require(cv::countNonZero(mask(cv::Rect(10, 0, 10, 20))) == 0,
                         "Bright cluster should stay background.");

    vision::SegmentationConfig offset_config = base_config;
    offset_config.threshold_offset = 50;
    segmenter = vision::makeKMeansSegmenter(offset_config);

    cv::Mat shifted(20, 20, CV_8UC1, cv::Scalar(160));
    shifted(cv::Rect(0, 0, 10, 20)).setTo(cv::Scalar(90));
    const cv::Mat offset_mask = segmenter->segment(shifted);

    vision_test::require(cv::countNonZero(offset_mask) == 400,
                         "Positive threshold offset should move both clusters into the foreground band.");
    return 0;
}
