#include "factories.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace vision {
namespace {

constexpr double kMinStdDev = 1e-6;

std::vector<double> computeStdDev(const std::vector<TrainingSample> &samples, std::size_t dimensions) {
    std::vector<double> mean(dimensions, 0.0);
    std::vector<double> stddev(dimensions, 1.0);

    int valid_count = 0;
    for (const TrainingSample &sample : samples) {
        if (sample.feature.values.size() != dimensions) {
            continue;
        }
        ++valid_count;
        for (std::size_t i = 0; i < dimensions; ++i) {
            mean[i] += sample.feature.values[i];
        }
    }

    if (valid_count <= 1) {
        return stddev;
    }

    for (double &value : mean) {
        value /= static_cast<double>(valid_count);
    }

    std::vector<double> variance(dimensions, 0.0);
    for (const TrainingSample &sample : samples) {
        if (sample.feature.values.size() != dimensions) {
            continue;
        }
        for (std::size_t i = 0; i < dimensions; ++i) {
            const double delta = sample.feature.values[i] - mean[i];
            variance[i] += delta * delta;
        }
    }

    for (std::size_t i = 0; i < dimensions; ++i) {
        const double sigma = std::sqrt(variance[i] / static_cast<double>(valid_count - 1));
        stddev[i] = (sigma < kMinStdDev) ? 1.0 : sigma;
    }
    return stddev;
}

class NearestNeighborClassifier final : public Classifier {
public:
    NearestNeighborClassifier(ClassificationConfig config, RunStorage &storage) : config_(std::move(config)), storage_(storage) {
        reload();
    }

    void reload() override {
        samples_ = storage_.loadTrainingSamples();
    }

    ClassifierResult predict(const FeatureVector &feature) const override {
        ClassifierResult result;
        if (feature.values.empty() || samples_.empty()) {
            return result;
        }

        const std::size_t dimensions = feature.values.size();
        const std::vector<double> stddev = computeStdDev(samples_, dimensions);
        double best_distance = std::numeric_limits<double>::infinity();
        std::string best_label = "Unknown";

        for (const TrainingSample &sample : samples_) {
            if (sample.feature.values.size() != dimensions) {
                continue;
            }

            double distance2 = 0.0;
            for (std::size_t i = 0; i < dimensions; ++i) {
                const double z = (feature.values[i] - sample.feature.values[i]) / stddev[i];
                distance2 += z * z;
            }

            const double distance = std::sqrt(distance2);
            if (distance < best_distance) {
                best_distance = distance;
                best_label = sample.label;
            }
        }

        if (!std::isfinite(best_distance)) {
            return result;
        }

        result.distance = best_distance;
        result.score = static_cast<float>(1.0 / (1.0 + best_distance));
        if (best_distance <= config_.unknown_threshold) {
            result.label = best_label;
        }
        return result;
    }

    bool train(const TrainingSample &sample) override {
        if (sample.label.empty() || sample.feature.values.empty()) {
            return false;
        }
        // Allocate the in-memory entry before committing to SQLite. If storage fails,
        // remove it so predictions and sampleCount still reflect committed records.
        samples_.push_back(sample);
        try {
            if (!storage_.appendTrainingSample(sample)) {
                samples_.pop_back();
                return false;
            }
        } catch (...) {
            samples_.pop_back();
            throw;
        }
        return true;
    }

    std::size_t sampleCount() const override {
        return samples_.size();
    }

private:
    ClassificationConfig config_;
    RunStorage &storage_;
    std::vector<TrainingSample> samples_;
};

}  // namespace

std::unique_ptr<Classifier> makeNearestNeighborClassifier(const ClassificationConfig &config, RunStorage &storage) {
    return std::make_unique<NearestNeighborClassifier>(config, storage);
}

}  // namespace vision
