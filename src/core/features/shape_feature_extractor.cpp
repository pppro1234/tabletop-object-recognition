#include "factories.hpp"

#include <cmath>
#include <memory>

namespace vision {
namespace {

double signedLog10(double value) {
    const double magnitude = std::log10(std::abs(value) + 1e-30);
    return (value < 0.0) ? -magnitude : magnitude;
}

class ShapeFeatureExtractor final : public FeatureExtractor {
public:
    FeatureVector extract(const Region &region) const override {
        FeatureVector feature;
        feature.extractor = "shape4d";
        feature.values = {
            region.fill_ratio,
            region.aspect_ratio,
            signedLog10(region.hu1),
            signedLog10(region.hu2),
        };
        return feature;
    }
};

}  // namespace

std::unique_ptr<FeatureExtractor> makeShapeFeatureExtractor() {
    return std::make_unique<ShapeFeatureExtractor>();
}

}  // namespace vision
