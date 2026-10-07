#include "factories.hpp"
#include "test_support.hpp"

#include <memory>

int main() {
    vision::AppConfig config = vision_test::makeTestConfig("unit_classifier_distance");

    std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);

    vision::ClassificationConfig classifier_config;
    classifier_config.unknown_threshold = 0.25;
    std::unique_ptr<vision::Classifier> classifier = vision::makeNearestNeighborClassifier(classifier_config, *storage);

    vision::TrainingSample first;
    first.label = "box";
    first.feature.values = {0.0, 0.0};

    vision::TrainingSample second;
    second.label = "rod";
    second.feature.values = {10.0, 10.0};

    vision_test::require(classifier->train(first), "Classifier should accept the first training sample.");
    vision_test::require(classifier->train(second), "Classifier should accept the second training sample.");
    vision_test::require(classifier->sampleCount() == 2, "Classifier should keep both training samples in memory.");

    vision::FeatureVector query;
    query.values = {1.0, 1.0};
    const vision::ClassifierResult prediction = classifier->predict(query);

    vision_test::require(prediction.label == "box", "Nearest standardized distance should prefer the closer sample.");
    vision_test::requireNear(prediction.distance, 0.2, 0.03, "Standardized Euclidean distance should match expectation.");
    vision_test::requireNear(prediction.score, 1.0 / 1.2, 0.03, "Score mapping should be based on 1 / (1 + distance).");

    vision::ClassificationConfig strict_config;
    strict_config.unknown_threshold = 0.15;
    classifier = vision::makeNearestNeighborClassifier(strict_config, *storage);
    const vision::ClassifierResult strict_prediction = classifier->predict(query);
    vision_test::require(strict_prediction.label == "Unknown",
                         "Prediction should fall back to Unknown when the best distance exceeds the threshold.");
    return 0;
}
