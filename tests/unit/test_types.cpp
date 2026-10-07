#include "vision/types.hpp"

#include <cassert>

int main() {
    vision::Detection detection;
    detection.track_id = 7;
    detection.label = "cube";
    detection.score = 0.92f;
    detection.centroid = cv::Point2f(10.0f, 15.0f);
    detection.angle_deg = 42.0f;

    assert(detection.track_id == 7);
    assert(detection.label == "cube");
    assert(detection.score > 0.9f);
    assert(detection.centroid.x == 10.0f);
    assert(detection.angle_deg == 42.0f);

    vision::FeatureVector feature;
    assert(feature.extractor == "shape4d");
    feature.values = {1.0, 2.0, 3.0, 4.0};
    assert(feature.values.size() == 4);

    vision::AppConfig config;
    assert(config.input.mode == vision::InputMode::kWebcam);
    assert(config.region.max_regions == 5);
    assert(config.storage.database_path.find("vision.sqlite3") != std::string::npos);
    return 0;
}
